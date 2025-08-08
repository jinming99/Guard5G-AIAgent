import threading
import logging
import time
import json
from .mobiflow import *
from .encoding import *
from .constant import *

class FactBase:

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.facts = {}
            self.bs_id_counter = 0
            self.bs_name_map = {}
            self.initialized = True
            # FBS DETECTION ADDITIONS
            self.fbs_detector = FBSDetector()  # Instance of FBS detector
            self.historical_measurements = {}  # Store historical data per UE
            self.known_good_cells = set()  # Whitelist of legitimate cells
            self.suspicious_cells = set()  # Blacklist of suspicious cells
            self.sdl_manager = None  # Will be set by main app

    def __new__(cls, *args, **kwargs):
        # singleton class
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
    
    def set_sdl_manager(self, sdl_manager):
        """Set SDL manager for storing alerts"""
        self.sdl_manager = sdl_manager

    def update_mobiflow(self) -> list:
        mf_list = []
        while True:
            write_should_end = True
            for ue in self.get_all_ue():
                if ue.should_report:
                    write_should_end = False
                    # generate UE mobiflow record
                    umf, prev_rrc, prev_nas, prev_sec, rrc, nas, sec = ue.generate_mobiflow()
                    # FBS DETECTION: Update FBS indicators
                    umf.update_fbs_indicators()
                    mf_list.append(umf)
                    # update BS
                    bs = self.get_bs(umf.bs_id)
                    if bs is not None:
                        bs.update_counters(prev_rrc, prev_nas, prev_sec, rrc, nas, sec)
                        # Check if this UE shows FBS indicators
                        if umf.suspected_fbs:
                            bs.suspicious_activity_count += 1
            for bs in self.get_all_bs():
                if bs.should_report:
                    write_should_end = False
                    # generate BS mobiflow record
                    bmf = bs.generate_mobiflow()
                    mf_list.append(bmf)
            if write_should_end:  # end writing if no mobiflow record to update
                break
        return mf_list

    def add_bs(self, bs: BS):
        with self._lock:
            if not self.facts.keys().__contains__(bs.bs_id):
                bs.ts = time.time() * 1000
                bs.bs_id = self.bs_id_counter
                self.bs_name_map[bs.name] = bs.bs_id
                self.bs_id_counter += 1
                self.facts[bs.bs_id] = bs
                # FBS DETECTION: Check if this is a known good cell
                if bs.cell_id and bs.mcc and bs.mnc:
                    cell_identity = f"{bs.mcc}:{bs.mnc}:{bs.cell_id}"
                    # You can populate known_good_cells from a config file
                    # For now, we'll treat the first cells as legitimate
                    if len(self.known_good_cells) < 5:
                        self.known_good_cells.add(cell_identity)
            else:
                self.facts[bs.bs_id].ts = time.time()

    def add_ue(self, bsId, ue: UE):
        with self._lock:
            if not self.facts.keys().__contains__(bsId):
                print("[Error] BS id not exist when trying to add UE: %s" % bsId)
                return
            else:
                ue.ts = time.time() * 1000
                ue.bs_id = bsId
                if self.facts[bsId] is not None:
                    self.facts[bsId].add_ue(ue)

    def get_bs_index_by_name(self, bs_name: str):
        if self.bs_name_map.keys().__contains__(bs_name):
            return self.bs_name_map[bs_name]
        else:
            return -1

    # BS related operations
    def get_all_bs(self):
        bss = []
        for bsId in self.facts.keys():
            bss.append(self.facts[bsId])
        return bss

    def get_bs(self, bsId):
        return self.facts.get(bsId)

    def remove_bs(self, bsId):
        if bsId in self.facts.keys():
            del self.facts[bsId]
            return True
        else:
            return False

    # UE related operations
    def get_all_ue(self):
        ues = []
        for bsId in self.facts.keys():
            ues += self.facts[bsId].ue
        return ues

    def get_ue(self, rnti: int):
        ues = self.get_all_ue()
        for ue in ues:
            if ue.rnti == rnti:
                return ue
        return None

    def remove_ue(self, rnti: int):
        ue_to_remove = None
        for bsId in self.facts.keys():
            for ue in self.facts[bsId].ue:
                if ue.rnti == rnti:
                    ue_to_remove = ue
                    break
            if ue_to_remove is not None:
                self.facts[bsId].ue.remove(ue_to_remove)
                return True
        return False

    def update_fact_base(self, kpm_measurement_dict: dict, me_id: str):
        if int(kpm_measurement_dict["UE.RNTI"]) == 0:
            return  # ignore empty indication records

        logging.info(f"[FactBase] KPM indication reported metrics: {kpm_measurement_dict}")
        
        # update fact base
        ue = UE()
        ue.rnti = int(kpm_measurement_dict["UE.RNTI"])
        ue.imsi = int(kpm_measurement_dict["UE.IMSI1"]) + (int(kpm_measurement_dict["UE.IMSI2"]) << 32)
        ue.tmsi = int(kpm_measurement_dict["UE.M_TMSI"])
        ue.rat = int(kpm_measurement_dict["UE.RAT"])
        ue.cipher_alg = int(kpm_measurement_dict["UE.CIPHER_ALG"])
        ue.integrity_alg = int(kpm_measurement_dict["UE.INTEGRITY_ALG"])
        ue.emm_cause = int(kpm_measurement_dict["UE.EMM_CAUSE"])
        
        # FBS DETECTION: Check for suspicious cipher algorithms
        if ue.cipher_alg == 0:  # NULL cipher
            ue.cipher_downgrade = True
            logging.warning(f"[FBS Detection] NULL cipher detected for UE {ue.rnti}")
        
        # FBS DETECTION: Track EMM cause codes
        if ue.emm_cause in [3, 6, 7, 8, 11, 12, 13, 14, 15]:  # Authentication/security failures
            ue.auth_reject_count += 1
            ue.last_auth_failure_time = time.time()
            logging.warning(f"[FBS Detection] Authentication failure for UE {ue.rnti}, cause: {ue.emm_cause}")
        
        # Process message trace
        msg_len = 20
        for i in range(1, msg_len+1):
            msg_val = int(kpm_measurement_dict[f"msg{i}"])
            if msg_val & 1 == 1:
                # RRC
                dcch = (msg_val >> 1) & 1
                downlink = (msg_val >> 2) & 1
                msg_id = (msg_val >> 3)
                msg_name = decode_rrc_msg(dcch, downlink, msg_id, ue.rat)
                if msg_name != "" and msg_name is not None:
                    ue.msg_trace.append(msg_name)
                    # FBS DETECTION: Track specific messages
                    self._analyze_rrc_message(ue, msg_name)
                elif msg_id != 0:
                    logging.error(f"[FactBase] Invalid RRC Msg dcch={dcch}, downlink={downlink} {msg_id}")
            else:
                # NAS
                dis = (msg_val >> 1) & 1
                msg_id = (msg_val >> 2)
                msg_name = decode_nas_msg(dis, msg_id, ue.rat)
                if msg_name != "" and msg_name is not None:
                    ue.msg_trace.append(msg_name)
                    # FBS DETECTION: Track specific messages
                    self._analyze_nas_message(ue, msg_name)
                elif msg_id != 0:
                    logging.error(f"[FactBase] Invalid NAS Msg: discriminator={dis} {msg_id}")
        
        # FBS DETECTION: Store signal measurements if available
        self._update_signal_measurements(ue, kpm_measurement_dict, me_id)
        
        # FBS DETECTION: Check for FBS indicators
        self._check_fbs_indicators(ue, me_id)
        
        # Add UE to fact base
        bs_id = self.get_bs_index_by_name(me_id)
        self.add_ue(bs_id, ue)
        
        # FBS DETECTION: If FBS detected, handle it
        if ue.suspected_fbs:
            self._handle_fbs_detection(ue, bs_id)
    
    def _analyze_rrc_message(self, ue: UE, msg_name: str):
        """Analyze RRC messages for FBS patterns"""
        # Track connection rejects
        if "Reject" in msg_name or "reject" in msg_name:
            ue.attach_failures += 1
            logging.info(f"[FBS Detection] Connection reject for UE {ue.rnti}")
        
        # Track rapid reselections
        if "Reselection" in msg_name or "Handover" in msg_name:
            ue.cell_reselection_count += 1
            ue.rapid_reselection_timestamp.append(time.time())
            if len(ue.rapid_reselection_timestamp) > 10:
                ue.rapid_reselection_timestamp.pop(0)
    
    def _analyze_nas_message(self, ue: UE, msg_name: str):
        """Analyze NAS messages for FBS patterns"""
        # Track authentication failures
        if "AUTHENTICATION_FAILURE" in msg_name:
            ue.auth_reject_count += 1
            ue.last_auth_failure_time = time.time()
        
        # Track attach failures
        if "ATTACH_REJECT" in msg_name or "SERVICE_REJECT" in msg_name:
            ue.attach_failures += 1
        
        # Track identity requests without security
        if "IDENTITY_REQUEST" in msg_name and ue.sec_state == SecState.SEC_CONTEXT_NOT_EXIST:
            ue.unauthenticated_id = ue.bs_id
            logging.warning(f"[FBS Detection] Unauthenticated identity request for UE {ue.rnti}")
        
        # Track location updates
        if "TRACKING_AREA_UPDATE" in msg_name:
            ue.unusual_lai_changes += 1
    
    def _update_signal_measurements(self, ue: UE, kpm_dict: dict, me_id: str):
        """Update signal measurements and check for anomalies"""
        # Store measurements for this UE
        if ue.rnti not in self.historical_measurements:
            self.historical_measurements[ue.rnti] = {
                'rsrp': [],
                'rsrq': [],
                'sinr': [],
                'cells': []
            }
        
        # Get signal measurements if available (these fields might need adjustment based on actual KPM)
        rsrp = kpm_dict.get("UE.RSRP", -100)
        rsrq = kpm_dict.get("UE.RSRQ", -20)
        sinr = kpm_dict.get("UE.SINR", 0)
        
        history = self.historical_measurements[ue.rnti]
        
        # Store current measurements
        if rsrp != -100:  # Valid measurement
            history['rsrp'].append(rsrp)
            if len(history['rsrp']) > 20:
                history['rsrp'].pop(0)
            
            # Check for signal anomaly
            if len(history['rsrp']) > 5:
                avg_rsrp = sum(history['rsrp'][:-1]) / (len(history['rsrp']) - 1)
                if self.fbs_detector.check_signal_anomaly(rsrp, avg_rsrp):
                    ue.signal_anomaly = True
                    logging.warning(f"[FBS Detection] Signal anomaly detected for UE {ue.rnti}: RSRP={rsrp}, Avg={avg_rsrp}")
        
        # Track cell changes
        history['cells'].append(me_id)
        if len(history['cells']) > 10:
            history['cells'].pop(0)
        
        # Update UE's cell history
        if me_id not in ue.previous_cell_ids:
            ue.previous_cell_ids.append(me_id)
            if len(ue.previous_cell_ids) > 5:
                ue.previous_cell_ids.pop(0)
    
    def _check_fbs_indicators(self, ue: UE, me_id: str):
        """Check for FBS indicators and update UE status"""
        # Check for rapid reselections
        if self.fbs_detector.check_rapid_reselection(ue.rapid_reselection_timestamp):
            ue.signal_anomaly = True
            logging.info(f"[FBS Detection] Rapid reselections detected for UE {ue.rnti}")
        
        # Check if connecting to suspicious cell
        bs = self.get_bs(ue.bs_id)
        if bs:
            cell_identity = f"{bs.mcc}:{bs.mnc}:{bs.cell_id}"
            if cell_identity in self.suspicious_cells:
                ue.suspected_fbs = True
                logging.warning(f"[FBS Detection] UE {ue.rnti} connected to suspicious cell {cell_identity}")
            elif cell_identity not in self.known_good_cells and len(self.known_good_cells) > 0:
                # Unknown cell - potentially suspicious
                if ue.attach_failures > 0 or ue.auth_reject_count > 0:
                    self.suspicious_cells.add(cell_identity)
                    ue.suspected_fbs = True
                    logging.warning(f"[FBS Detection] New suspicious cell detected: {cell_identity}")
        
        # Update overall FBS indicators
        ue.update_fbs_indicators()
    
    def _handle_fbs_detection(self, ue: UE, bs_id: int):
        """Handle new FBS detection"""
        logging.warning(f"[FBS DETECTED] UE: {ue.rnti}, BS: {bs_id}")
        logging.warning(f"  Indicators: Attach failures={ue.attach_failures}, "
                       f"Auth rejects={ue.auth_reject_count}, "
                       f"Cipher downgrade={ue.cipher_downgrade}, "
                       f"Signal anomaly={ue.signal_anomaly}")
        
        # Mark cell as suspicious
        bs = self.get_bs(bs_id)
        if bs:
            cell_identity = f"{bs.mcc}:{bs.mnc}:{bs.cell_id}"
            self.suspicious_cells.add(cell_identity)
            bs.suspicious_activity_count += 1
        
        # Update SDL with FBS detection count if SDL manager is available
        if self.sdl_manager:
            try:
                self.sdl_manager.increment_fbs_detection_count()
                # Store alert in SDL
                self._store_fbs_alert(ue, bs_id)
            except Exception as e:
                logging.error(f"Failed to update SDL: {e}")
        
        # Trigger alert/notification
        self._trigger_fbs_alert(ue, bs_id)
    
    def _store_fbs_alert(self, ue: UE, bs_id: int):
        """Store FBS alert in SDL"""
        if not self.sdl_manager:
            return
        
        alert = {
            'type': 'FBS_DETECTION',
            'severity': 'HIGH',
            'timestamp': time.time(),
            'ue_rnti': ue.rnti,
            'ue_tmsi': ue.tmsi,
            'bs_id': bs_id,
            'indicators': {
                'attach_failures': ue.attach_failures,
                'auth_reject_count': ue.auth_reject_count,
                'cipher_downgrade': ue.cipher_downgrade,
                'signal_anomaly': ue.signal_anomaly,
                'unexpected_redirect': ue.unexpected_redirect,
                'cell_reselection_count': ue.cell_reselection_count
            }
        }
        
        # Store alert in SDL
        alert_key = f"alert:fbs:{ue.rnti}:{int(time.time())}"
        self.sdl_manager.store_data_to_sdl("alerts", alert_key, json.dumps(alert))
    
    def _trigger_fbs_alert(self, ue: UE, bs_id: int):
        """Trigger FBS detection alert (can be extended to send notifications)"""
        # This can be extended to send notifications to external systems
        # For now, just log the alert
        alert_msg = f"FBS ALERT: UE {hex(ue.rnti)} potentially under FBS attack from BS {bs_id}"
        logging.critical(alert_msg)
        
        # You can add more alert mechanisms here:
        # - Send SNMP trap
        # - Post to webhook
        # - Send to monitoring system
        # - Trigger automated response