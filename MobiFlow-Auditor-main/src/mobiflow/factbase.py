import threading
import logging
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

    def __new__(cls, *args, **kwargs):
        # singleton class
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def update_mobiflow(self) -> list:
        mf_list = []
        while True:
            write_should_end = True
            for ue in self.get_all_ue():
                if ue.should_report:
                    write_should_end = False
                    # generate UE mobiflow record
                    umf, prev_rrc, prev_nas, prev_sec, rrc, nas, sec = ue.generate_mobiflow()
                    mf_list.append(umf)
                    # update BS
                    bs = self.get_bs(umf.bs_id)
                    if bs is not None:
                        bs.update_counters(prev_rrc, prev_nas, prev_sec, rrc, nas, sec)
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
                elif msg_id != 0:
                    logging.error(f"[FactBase] Invalid RRC Msg dcch={dcch}, downlink={downlink} {msg_id}")
            else:
                # NAS
                dis = (msg_val >> 1) & 1
                msg_id = (msg_val >> 2)
                msg_name = decode_nas_msg(dis, msg_id, ue.rat)
                if msg_name != "" and msg_name is not None:
                    ue.msg_trace.append(msg_name)
                elif msg_id != 0:
                    logging.error(f"[FactBase] Invalid NAS Msg: discriminator={dis} {msg_id}")

        self.add_ue(self.get_bs_index_by_name(me_id), ue)
        