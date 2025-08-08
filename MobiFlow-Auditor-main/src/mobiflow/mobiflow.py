import time
from enum import Enum


def get_time_ms():
    return int(time.time() * 1000)

###################### Auxiliary classes ######################

class State(Enum):
    def __str__(self):
        return str(self.value)

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

class RRCState(State):
    INACTIVE = 0
    RRC_IDLE = 1
    RRC_CONNECTED = 2
    RRC_RECONGIFURED = 3


class EMMState(State):
    EMM_DEREGISTERED = 0
    EMM_REGISTER_INIT = 1
    EMM_REGISTERED = 2


class SecState(State):
    SEC_CONTEXT_NOT_EXIST = 0
    SEC_CONTEXT_EXIST = 1

###################### Constants ######################

MOBIFLOW_VERSION = "v1.0"
GENERATOR_NAME = "SECSM"
UE_MOBIFLOW_ID_COUNTER = 0
BS_MOBIFLOW_ID_COUNTER = 0
MOBIFLOW_DELIMITER = ";"

###################### MobiFlow Structures ######################

class UEMobiFlow:
    def __init__(self):
        self.msg_type = "UE"            # Msg hdr  - mobiflow type [UE, BS]
        self.msg_id = 0                 # Msg hdr  - unique mobiflow event ID
        self.timestamp = get_time_ms()              # Msg hdr  - timestamp (ms)
        self.mobiflow_ver = MOBIFLOW_VERSION        # Msg hdr  - version of Mobiflow
        self.generator_name = GENERATOR_NAME        # Msg hdr  - generator name (e.g., SECSM)
        #####################################################################
        self.bs_id = 0                  # UE meta  - basestation id
        self.rnti = 0                   # UE meta  - ue rnti
        self.tmsi = 0                   # UE meta  - ue mtmsi
        self.imsi = ""                  # UE meta  - ue imsi
        self.imei = ""                  # UE meta  - ue imei
        #####################################################################
        self.cipher_alg = 0             # UE stats  - cipher algorithm
        self.integrity_alg = 0          # UE stats  - integrity algorithm
        self.establish_cause = 0        # UE stats  - establishment cause
        self.msg = ""                   # UE stats  - RRC/NAS message
        self.rrc_state = 0              # UE stats  - RRC state       [INACTIVE, RRC_IDLE, RRC_CONNECTED, RRC_RECONFIGURED]
        self.nas_state = 0              # UE stats  - NAS state (EMM) [EMM_DEREGISTERED, EMM_REGISTER_INIT, EMM_REGISTERED]
        self.sec_state = 0              # UE stats  - security state  [SEC_CONTEXT_NOT_EXIST, SEC_CONTEXT_EXIST]
        self.emm_cause = 0              # UE stats  - EMM cause code
        #####################################################################
        self.rrc_initial_timer = 0      # UE timer  -
        self.rrc_inactive_timer = 0     # UE timer  -
        self.nas_initial_timer = 0      # UE timer  -
        self.nas_inactive_timer = 0     # UE timer  -
        #####################################################################
        # FBS DETECTION FIELDS - NEW
        #####################################################################
        self.suspected_fbs = False      # FBS Detection - Boolean flag for suspected fake base station
        self.attach_failures = 0        # FBS Detection - Count of consecutive attach failures
        self.unauthenticated_id = None  # FBS Detection - Unauthenticated cell/network ID if detected
        self.cell_reselection_count = 0 # FBS Detection - Rapid cell reselections
        self.auth_reject_count = 0      # FBS Detection - Authentication rejection count
        self.unusual_lai_changes = 0    # FBS Detection - Location Area Identity changes
        self.cipher_downgrade = False   # FBS Detection - Detected cipher algorithm downgrade
        self.unexpected_redirect = False # FBS Detection - Unexpected network redirect
        self.signal_anomaly = False     # FBS Detection - Signal strength anomaly detected

    def __str__(self):
        attrs = []
        for a in self.__dict__.values():
            if str(a) == "":
                attrs.append(" ")  # avoid parse error in C
            else:
                attrs.append(str(a))
        return MOBIFLOW_DELIMITER.join(attrs)
    
    def to_dict(self):
        """
        Convert MobiFlow record to dictionary for JSON serialization
        """
        return {
            'msg_type': self.msg_type,
            'msg_id': self.msg_id,
            'timestamp': self.timestamp,
            'mobiflow_ver': self.mobiflow_ver,
            'generator_name': self.generator_name,
            'bs_id': self.bs_id,
            'rnti': self.rnti,
            'tmsi': self.tmsi,
            'imsi': self.imsi,
            'imei': self.imei,
            'cipher_alg': self.cipher_alg,
            'integrity_alg': self.integrity_alg,
            'establish_cause': self.establish_cause,
            'msg': self.msg,
            'rrc_state': self.rrc_state,
            'nas_state': self.nas_state,
            'sec_state': self.sec_state,
            'emm_cause': self.emm_cause,
            'rrc_initial_timer': self.rrc_initial_timer,
            'rrc_inactive_timer': self.rrc_inactive_timer,
            'nas_initial_timer': self.nas_initial_timer,
            'nas_inactive_timer': self.nas_inactive_timer,
            # FBS Detection fields
            'suspected_fbs': self.suspected_fbs,
            'attach_failures': self.attach_failures,
            'unauthenticated_id': self.unauthenticated_id,
            'cell_reselection_count': self.cell_reselection_count,
            'auth_reject_count': self.auth_reject_count,
            'unusual_lai_changes': self.unusual_lai_changes,
            'cipher_downgrade': self.cipher_downgrade,
            'unexpected_redirect': self.unexpected_redirect,
            'signal_anomaly': self.signal_anomaly
        }
    
    def update_fbs_indicators(self):
        """
        Update FBS detection based on current indicators
        Call this after updating any relevant fields
        """
        fbs_score = 0
        
        # Calculate FBS likelihood score
        if self.attach_failures > 2:
            fbs_score += 2
        if self.auth_reject_count > 0:
            fbs_score += 3
        if self.cell_reselection_count > 5:
            fbs_score += 1
        if self.cipher_downgrade:
            fbs_score += 3
        if self.unexpected_redirect:
            fbs_score += 2
        if self.signal_anomaly:
            fbs_score += 1
        if self.unusual_lai_changes > 2:
            fbs_score += 1
        
        # Set suspected_fbs flag if score exceeds threshold
        self.suspected_fbs = (fbs_score >= 4)
        
        # Record first suspicion time
        if self.suspected_fbs and not hasattr(self, 'fbs_first_suspected_time'):
            self.fbs_first_suspected_time = time.time()
        
        return self.suspected_fbs

class BSMobiFlow:
    def __init__(self):
        self.msg_type = "BS"            # Msg hdr  - mobiflow type [UE, BS]
        self.msg_id = 0                 # Msg hdr  - unique mobiflow event ID
        self.timestamp = get_time_ms()              # Msg hdr  - timestamp (ms)
        self.mobiflow_ver = MOBIFLOW_VERSION        # Msg hdr  - version of Mobiflow
        self.generator_name = GENERATOR_NAME        # Msg hdr  - generator name (e.g., SECSM)
        #####################################################################
        self.bs_id = 0                  # BS meta  - basestation id
        self.mcc = ""                   # BS meta  - mobile country code
        self.mnc = ""                   # BS meta  - mobile network code
        self.tac = ""                   # BS meta  - tracking area code
        self.cell_id = ""               # BS meta  - cell Id
        self.report_period = 0          # BS meta  - report period (ms)
        #####################################################################
        self.connected_ue_cnt = 0       # BS stats -
        self.idle_ue_cnt = 0            # BS stats -
        self.max_ue_cnt = 0             # BS stats -
        #####################################################################
        self.initial_timer = 0          # BS timer  -
        self.inactive_timer = 0         # BS timer  -
        #####################################################################
        # FBS DETECTION FIELDS - NEW
        #####################################################################
        self.suspicious_activity_count = 0  # FBS Detection - Count of suspicious activities
        self.mass_reject_count = 0          # FBS Detection - Mass rejection events
        self.abnormal_release_count = 0     # FBS Detection - Abnormal connection releases

    def __str__(self):
        attrs = []
        for a in self.__dict__.values():
            if str(a) == "":
                attrs.append(" ")  # avoid parse error in C
            else:
                attrs.append(str(a))
        return MOBIFLOW_DELIMITER.join(attrs)
    
    def to_dict(self):
        """
        Convert BS MobiFlow record to dictionary for JSON serialization
        """
        return {
            'msg_type': self.msg_type,
            'msg_id': self.msg_id,
            'timestamp': self.timestamp,
            'mobiflow_ver': self.mobiflow_ver,
            'generator_name': self.generator_name,
            'bs_id': self.bs_id,
            'mcc': self.mcc,
            'mnc': self.mnc,
            'tac': self.tac,
            'cell_id': self.cell_id,
            'report_period': self.report_period,
            'connected_ue_cnt': self.connected_ue_cnt,
            'idle_ue_cnt': self.idle_ue_cnt,
            'max_ue_cnt': self.max_ue_cnt,
            'initial_timer': self.initial_timer,
            'inactive_timer': self.inactive_timer,
            # FBS Detection fields
            'suspicious_activity_count': self.suspicious_activity_count,
            'mass_reject_count': self.mass_reject_count,
            'abnormal_release_count': self.abnormal_release_count
        }


###################### SECSM Structures ######################

class UE:
    def __init__(self) -> None:
        #### UE Meta ####
        self.bs_id = 0
        self.rnti = 0
        self.tmsi = 0
        self.imsi = ""
        self.imei = 0
        #### UE Attributes ####
        self.rat = -1
        self.cipher_alg = 0
        self.integrity_alg = 0
        self.establish_cause = 0
        self.rrc_state = RRCState.INACTIVE
        self.nas_state = EMMState.EMM_DEREGISTERED
        self.sec_state = SecState.SEC_CONTEXT_NOT_EXIST
        self.emm_cause = 0
        #### UE Timer ####
        self.rrc_initial_timer = 0
        self.rrc_inactive_timer = 0
        self.nas_initial_timer = 0
        self.nas_inactive_timer = 0
        #### History record ####
        self.msg_trace = []
        self.current_msg_index = 0    # indicate how many messages are reported to PBEST
        # UE mobiflow report criteria: report new msg
        self.should_report = True     # indicate whether the UE state has changed and should be reported
        self.last_ts = 0              # indicate timestamp last time the UE is updated
        #### FBS DETECTION FIELDS - NEW ####
        self.suspected_fbs = False
        self.attach_failures = 0
        self.auth_reject_count = 0
        self.unauthenticated_id = None
        self.cell_reselection_count = 0
        self.unusual_lai_changes = 0
        self.cipher_downgrade = False
        self.unexpected_redirect = False
        self.signal_anomaly = False
        self.rapid_reselection_timestamp = []
        self.last_auth_failure_time = None
        self.fbs_first_suspected_time = None
        self.previous_cell_ids = []
        self.previous_plmns = []

    def update(self, ur) -> bool:
        if isinstance(ur, UE):
            if len(ur.msg_trace) != 0:
                self.msg_trace = self.msg_trace + ur.msg_trace
                self.should_report = True
            if self.tmsi != ur.tmsi:
                self.tmsi = ur.tmsi
                self.should_report = True
            if self.establish_cause != ur.establish_cause:
                self.establish_cause = ur.establish_cause
                self.should_report = True
            if self.cipher_alg != ur.cipher_alg:
                # Check for cipher downgrade
                if ur.cipher_alg == 0 and self.cipher_alg != 0:
                    self.cipher_downgrade = True
                self.cipher_alg = ur.cipher_alg
                self.should_report = True
            if self.integrity_alg != ur.integrity_alg:
                self.integrity_alg = ur.integrity_alg
                self.should_report = True
            # Update FBS detection fields
            if hasattr(ur, 'attach_failures'):
                self.attach_failures = ur.attach_failures
            if hasattr(ur, 'auth_reject_count'):
                self.auth_reject_count = ur.auth_reject_count
            return True
        else:
            return False

    def __str__(self) -> str:
        s = "UE: {rnti: %s, tmsi: %s, imsi: %s, imei: %s}\n" % (hex(self.rnti), hex(self.tmsi), self.imsi, self.imei)
        m = "MsgTrace: \n"
        for msg in self.msg_trace:
            m = m + msg + "\n"
        return s + m

    def __eq__(self, other) -> bool:
        if not isinstance(other, UE):
            return False
        return self.bs_id == other.bs_id and self.rnti == other.rnti

    def propagate(self, msg: str):
        # implement state transition and counter update
        prev_rrc_state = self.rrc_state
        prev_nas_state = self.nas_state
        prev_sec_state = self.sec_state
        
        # FBS Detection: Track message for patterns
        if msg == "RRCConnectionReject" or msg == "RRCConnectionReestablishmentReject" or msg == "RRCReject":
            self.attach_failures += 1
            self.last_auth_failure_time = time.time()
            self.rrc_state = RRCState.RRC_IDLE
            
        elif msg == "AUTHENTICATION_FAILURE":
            self.auth_reject_count += 1
            self.last_auth_failure_time = time.time()
            
        elif msg == "RRCConnectionResume-r13" or msg == "RRCConnectionSetup" or msg == "RRCConnectionReestablishment"\
                or msg == "RRCResume" or msg == "RRCSetup" or msg == "RRCReestablishment":
            if self.rrc_state == RRCState.INACTIVE or self.rrc_state == RRCState.RRC_IDLE:
                self.rrc_state = RRCState.RRC_CONNECTED
                # Track rapid reselections
                self.rapid_reselection_timestamp.append(time.time())
                if len(self.rapid_reselection_timestamp) > 10:
                    self.rapid_reselection_timestamp.pop(0)

        elif msg == "RRCConnectionRelease" or msg == "RRCRelease":
            self.rrc_state = RRCState.RRC_IDLE
            self.nas_state = EMMState.EMM_DEREGISTERED
            self.sec_state = SecState.SEC_CONTEXT_NOT_EXIST

        elif msg == "ATTACH_REQUEST" or msg == "RRCConnectionReestablishmentComplete"\
                or msg == "Registrationrequest" or msg == "Servicerequest":
            self.nas_state = EMMState.EMM_REGISTER_INIT

        elif msg == "SERVICE_REQUEST": # LTE: The UE shall treat that the service request procedure as successful, when the lower layers indicate that the user plane radio bearer is successfully set up
            self.nas_state = EMMState.EMM_REGISTERED

        elif msg == "SecurityModeComplete": # RRC security state
                # or msg == "Securitymodecomplete":
            self.sec_state = SecState.SEC_CONTEXT_EXIST

        elif msg == "ATTACH_COMPLETE"\
                or msg == "RRCReconfigurationComplete" or msg == "Registrationcomplete" or msg == "Serviceaccept":
            self.nas_state = EMMState.EMM_REGISTERED
            # Reset failure counters on successful attach
            self.attach_failures = 0
            self.auth_reject_count = 0

        elif msg == "ATTACH_REJECT" or msg == "SERVICE_REJECT" or msg == "DETACH_ACCEPT" or msg == "TRACKING_AREA_UPDATE_REJECT" \
                or msg == "Registrationreject" or msg == "Servicereject" or msg == "DeregistrationacceptUEoriginating":
            self.nas_state = EMMState.EMM_DEREGISTERED
            self.sec_state = SecState.SEC_CONTEXT_NOT_EXIST
            self.attach_failures += 1
            
        elif msg == "IDENTITY_REQUEST":
            # Check if unauthenticated
            if self.sec_state == SecState.SEC_CONTEXT_NOT_EXIST:
                self.unauthenticated_id = self.bs_id
                
        elif msg == "TRACKING_AREA_UPDATE_REQUEST":
            self.unusual_lai_changes += 1

        # update timer
        if prev_rrc_state != self.rrc_state:
            if prev_rrc_state < RRCState.RRC_CONNECTED <= self.rrc_state:
                self.rrc_initial_timer = self.last_ts
            if self.rrc_state < RRCState.RRC_CONNECTED <= prev_rrc_state:
                self.rrc_inactive_timer = self.last_ts
                self.cell_reselection_count += 1
        if prev_nas_state != self.nas_state:
            if prev_nas_state <= EMMState.EMM_DEREGISTERED < self.nas_state:
                self.nas_initial_timer = self.last_ts
            if self.nas_state <= EMMState.EMM_DEREGISTERED < prev_nas_state or prev_nas_state < EMMState.EMM_REGISTERED <= self.nas_state:
                self.nas_inactive_timer = self.last_ts
        if prev_sec_state != self.sec_state:
            pass

        # Update FBS indicators after state changes
        self.update_fbs_indicators()

        return prev_rrc_state, prev_nas_state, prev_sec_state, self.rrc_state, self.nas_state, self.sec_state

    def update_fbs_indicators(self):
        """
        Update FBS detection based on current indicators
        """
        fbs_score = 0
        
        # Calculate FBS likelihood score
        if self.attach_failures > 2:
            fbs_score += 2
        if self.auth_reject_count > 0:
            fbs_score += 3
        if self.cell_reselection_count > 5:
            fbs_score += 1
        if self.cipher_downgrade:
            fbs_score += 3
        if self.unexpected_redirect:
            fbs_score += 2
        if self.signal_anomaly:
            fbs_score += 1
        if self.unusual_lai_changes > 2:
            fbs_score += 1
        
        # Check for rapid reselections
        if len(self.rapid_reselection_timestamp) >= 3:
            recent = [t for t in self.rapid_reselection_timestamp 
                     if (time.time() - t) < 30]  # Within 30 seconds
            if len(recent) >= 3:
                fbs_score += 2
        
        # Set suspected_fbs flag if score exceeds threshold
        self.suspected_fbs = (fbs_score >= 4)
        
        # Record first suspicion time
        if self.suspected_fbs and not self.fbs_first_suspected_time:
            self.fbs_first_suspected_time = time.time()
        
        return self.suspected_fbs

    def generate_mobiflow(self):
        umf = UEMobiFlow()
        # propagate constant
        global UE_MOBIFLOW_ID_COUNTER
        umf.msg_id = UE_MOBIFLOW_ID_COUNTER
        UE_MOBIFLOW_ID_COUNTER += 1
        umf.timestamp = get_time_ms()
        umf.bs_id = self.bs_id
        umf.rnti = self.rnti
        umf.tmsi = self.tmsi
        umf.imsi = self.imsi
        umf.imei = self.imei
        umf.cipher_alg = self.cipher_alg
        umf.integrity_alg = self.integrity_alg
        umf.establish_cause = self.establish_cause
        umf.emm_cause = self.emm_cause
        # Copy FBS detection fields
        umf.suspected_fbs = self.suspected_fbs
        umf.attach_failures = self.attach_failures
        umf.auth_reject_count = self.auth_reject_count
        umf.unauthenticated_id = self.unauthenticated_id
        umf.cell_reselection_count = self.cell_reselection_count
        umf.unusual_lai_changes = self.unusual_lai_changes
        umf.cipher_downgrade = self.cipher_downgrade
        umf.unexpected_redirect = self.unexpected_redirect
        umf.signal_anomaly = self.signal_anomaly
        
        prev_rrc, prev_nas, prev_sec, rrc, nas, sec = 0, 0, 0, 0, 0, 0
        if self.current_msg_index < self.msg_trace.__len__():
            umf.msg = self.msg_trace[self.current_msg_index]
            self.current_msg_index += 1
            prev_rrc, prev_nas, prev_sec, rrc, nas, sec = self.propagate(umf.msg)  # update UE state
            umf.rrc_state = self.rrc_state
            umf.nas_state = self.nas_state
            umf.sec_state = self.sec_state
            umf.rrc_initial_timer = self.rrc_initial_timer
            umf.rrc_inactive_timer = self.rrc_inactive_timer
            umf.nas_initial_timer = self.nas_initial_timer
            umf.nas_inactive_timer = self.nas_inactive_timer
        if self.current_msg_index >= self.msg_trace.__len__():
            self.should_report = False
        return umf, prev_rrc, prev_nas, prev_sec, rrc, nas, sec


class BS:
    def __init__(self) -> None:
        ####  BS Meta ####
        self.bs_id = 0
        self.mcc = 0
        self.mnc = 0
        self.tac = 0
        self.cell_id = 0
        self.report_period = 0
        #### BS Stats ####
        self.connected_ue_cnt = 0
        self.idle_ue_cnt = 0
        self.max_ue_cnt = 0
        #### BS Timer ####
        self.initial_timer = get_time_ms()
        self.inactive_timer = 0
        #### History record ####
        self.ue = []
        # BS mobiflow report criteria: change of BS stats / timer
        self.should_report = True
        self.name = ""
        #### FBS DETECTION FIELDS - NEW ####
        self.suspicious_activity_count = 0
        self.mass_reject_count = 0
        self.abnormal_release_count = 0
        self.fbs_ue_count = 0  # Count of UEs suspected of FBS attack

    def add_ue(self, ur: UE) -> None:
        # update UE if found
        for u in self.ue:
            if u.__eq__(ur):
                u.last_ts = get_time_ms()
                u.update(ur)
                # Check if UE is now suspected of FBS
                if u.suspected_fbs and not ur.suspected_fbs:
                    self.fbs_ue_count += 1
                    self.suspicious_activity_count += 1
                return

        # add new UE
        ur.last_ts = get_time_ms()
        self.ue.append(ur)
        if ur.suspected_fbs:
            self.fbs_ue_count += 1

    def __str__(self) -> str:
        uestr = ""
        for u in self.ue:
            uestr = uestr + hex(u.rnti) + " "
        s = "BsRecord: {%d}" % self.bs_id
        return s

    def __eq__(self, other) -> bool:
        if not isinstance(other, BS):
            return False
        return self.bs_id == other.bs_id

    def update_counters(self, prev_rrc, prev_nas, prev_sec, rrc, nas, sec):
        if prev_rrc != rrc:
            if rrc == RRCState.RRC_CONNECTED:
                self.connected_ue_cnt += 1
                self.should_report = True
            if rrc == RRCState.RRC_IDLE:
                self.idle_ue_cnt += 1
                self.should_report = True
            if prev_rrc == RRCState.RRC_CONNECTED and rrc < RRCState.RRC_CONNECTED:
                self.connected_ue_cnt -= 1
                self.abnormal_release_count += 1  # Track abnormal releases
            if prev_rrc == RRCState.RRC_IDLE and rrc > RRCState.RRC_IDLE:
                self.idle_ue_cnt -= 1
        if prev_nas != nas:
            self.should_report = True
            # Track mass rejections
            if nas == EMMState.EMM_DEREGISTERED and prev_nas != EMMState.EMM_DEREGISTERED:
                self.mass_reject_count += 1
        if prev_sec != sec:
            self.should_report = True
        if prev_rrc == rrc and prev_nas == nas and prev_sec == sec:
            self.should_report = False

    def generate_mobiflow(self) -> BSMobiFlow:
        bmf = BSMobiFlow()
        # propagate constant
        global BS_MOBIFLOW_ID_COUNTER
        bmf.msg_id = BS_MOBIFLOW_ID_COUNTER
        BS_MOBIFLOW_ID_COUNTER += 1
        bmf.timestamp = get_time_ms()
        bmf.bs_id = self.bs_id
        bmf.mcc = self.mcc
        bmf.mnc = self.mnc
        bmf.tac = self.tac
        bmf.cell_id = self.cell_id
        bmf.report_period = self.report_period
        bmf.connected_ue_cnt = self.connected_ue_cnt
        bmf.idle_ue_cnt = self.idle_ue_cnt
        bmf.max_ue_cnt = self.max_ue_cnt
        bmf.initial_timer = self.initial_timer
        bmf.inactive_timer = self.inactive_timer
        # Copy FBS detection fields
        bmf.suspicious_activity_count = self.suspicious_activity_count
        bmf.mass_reject_count = self.mass_reject_count
        bmf.abnormal_release_count = self.abnormal_release_count
        self.should_report = False
        return bmf


###################### FBS DETECTION HELPER CLASS - NEW ######################

class FBSDetector:
    """
    Helper class for FBS detection logic
    """
    
    @staticmethod
    def check_signal_anomaly(current_rsrp, historical_rsrp_avg, threshold=15):
        """
        Check for signal strength anomalies
        Fake base stations often have unusual signal patterns
        """
        if not historical_rsrp_avg:
            return False
        
        # Sudden strong signal from unknown cell
        if current_rsrp > historical_rsrp_avg + threshold:
            return True
        
        return False
    
    @staticmethod
    def check_rapid_reselection(reselection_times, window_seconds=30):
        """
        Check for rapid cell reselections
        FBS may cause frequent reselections
        """
        if len(reselection_times) < 3:
            return False
        
        # Check if 3+ reselections within window
        recent = [t for t in reselection_times if 
                  (time.time() - t) < window_seconds]
        
        return len(recent) >= 3
    
    @staticmethod
    def check_identity_mismatch(advertised_plmn, expected_plmns):
        """
        Check for PLMN/network identity mismatches
        """
        if not expected_plmns:
            return False
        
        return advertised_plmn not in expected_plmns