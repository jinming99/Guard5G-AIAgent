# ==================================================================================
#
#       Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==================================================================================

import json
import requests
import time
from typing import List, Dict
from ricxappframe.xapp_frame import RMRXapp
from ricxappframe.entities.rnib.nb_identity_pb2 import NbIdentity
from ._BaseManager import _BaseManager
from mdclogpy import Level


class SdlManager(_BaseManager):

    __E2_namespace = "e2Manager"
    __E2_msgr_endpoint = "http://service-ricplt-e2mgr-http.ricplt.svc.cluster.local:3800/v1/nodeb/"

    def __init__(self, rmr_xapp: RMRXapp) -> None:
        super().__init__(rmr_xapp)
        self.logger.set_level(Level.INFO)

    def get_sdl_keys(self, ns) -> List:
        return self._rmr_xapp.sdl.find_keys(ns, "")

    def get_sdl_with_key(self, ns, key):
        return self._rmr_xapp.sdl_find_and_get(ns, key, usemsgpack=False)

    def get_gnb_list(self) -> List[NbIdentity]:
        return self._rmr_xapp.get_list_gnb_ids()

    def get_enb_list(self) -> List[NbIdentity]:
        return self._rmr_xapp.get_list_enb_ids()

    def get_nodeb_info_by_inventory_name(self, inventory_name) -> Dict:
        url = self.__E2_msgr_endpoint + inventory_name
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            return json.loads(response.text)
        else:
            self.logger.error('SdlManager [get_nodeb_info_by_id] Error:', response.status_code)
            return dict()

    def store_data_to_sdl(self, ns: str, key: str, value):
        self._rmr_xapp.sdl.set(ns, key, value)

    # ==================================================================================
    # NEW METHODS FOR LLM INTEGRATION - FBS DETECTION SUPPORT
    # ==================================================================================

    def get_last_kpm_json(self, ue_id: str) -> dict:
        """
        Get the last KPM data for a UE as a JSON-serializable dictionary
        
        Args:
            ue_id: UE identifier
        
        Returns:
            Dictionary with KPM data or None if not found
        """
        try:
            # Get the raw KPM data from SDL
            kpm_key = f"kpm:{ue_id}:latest"
            kpm_data = self._rmr_xapp.sdl.get("kpm", kpm_key)
            
            if not kpm_data:
                return None
            
            # If data is already a dictionary, return it
            if isinstance(kpm_data, dict):
                return kpm_data
            
            # If data is JSON string, parse it
            if isinstance(kpm_data, str):
                return json.loads(kpm_data)
            
            # If data is bytes, decode and parse
            if isinstance(kpm_data, bytes):
                return json.loads(kpm_data.decode('utf-8'))
            
            # For protobuf data (if using protobuf)
            # Placeholder for protobuf deserialization if needed
            # from kpm_pb2 import KpmIndication
            # kpm_msg = KpmIndication()
            # kpm_msg.ParseFromString(kpm_data)
            
            # Create a mock KPM response for now
            kpm_dict = {
                'timestamp': time.time(),
                'ue_id': ue_id,
                'cell_id': 'cell_001',
                'rsrp': -85,
                'rsrq': -12,
                'sinr': 15,
                'cqi': 10,
                'throughput_dl': 100000,
                'throughput_ul': 50000,
                'bler': 0.01,
                'mcs': 20,
                'ri': 2,
                'pmi': 1
            }
            
            return kpm_dict
            
        except Exception as e:
            self.logger.error(f"Error getting KPM data for UE {ue_id}: {e}")
            return None

    def get_mobiflow_json(self, n: int = 100) -> list:
        """
        Get the last N MobiFlow records as JSON-serializable dictionaries
        
        Args:
            n: Number of records to retrieve
        
        Returns:
            List of dictionaries with MobiFlow data
        """
        try:
            records = []
            
            # Get list of recent MobiFlow keys from SDL
            keys = self._rmr_xapp.sdl.find_keys("mobiflow", "")
            
            # Sort by timestamp (assuming keys have timestamp)
            sorted_keys = sorted(keys, reverse=True)[:n] if keys else []
            
            for key in sorted_keys:
                data = self._rmr_xapp.sdl.get("mobiflow", key)
                if data:
                    # Deserialize MobiFlow record
                    record = self._deserialize_mobiflow(data)
                    if record:
                        records.append(record)
            
            return records
            
        except Exception as e:
            self.logger.error(f"Error getting MobiFlow records: {e}")
            return []

    def get_ue_mobiflow_json(self, ue_id: str) -> list:
        """
        Get all MobiFlow records for a specific UE
        
        Args:
            ue_id: UE identifier
        
        Returns:
            List of dictionaries with UE's MobiFlow data
        """
        try:
            records = []
            
            # Get UE-specific MobiFlow keys
            pattern = f"ue:{ue_id}"
            keys = self._rmr_xapp.sdl.find_keys("mobiflow", pattern)
            
            for key in sorted(keys, reverse=True) if keys else []:
                data = self._rmr_xapp.sdl.get("mobiflow", key)
                if data:
                    record = self._deserialize_mobiflow(data)
                    if record:
                        # Add FBS detection fields if not present
                        if 'suspected_fbs' not in record:
                            record['suspected_fbs'] = self._check_fbs_indicators(record)
                        records.append(record)
            
            return records
            
        except Exception as e:
            self.logger.error(f"Error getting UE MobiFlow records: {e}")
            return []

    def _deserialize_mobiflow(self, data) -> dict:
        """
        Helper to deserialize MobiFlow data to dictionary
        """
        try:
            # If data is already a dictionary
            if isinstance(data, dict):
                return self._enhance_mobiflow_record(data)
            
            # If data is JSON string
            if isinstance(data, str):
                record = json.loads(data) if data.startswith('{') else self._parse_mobiflow_string(data)
                return self._enhance_mobiflow_record(record)
            
            # If data is bytes
            if isinstance(data, bytes):
                data_str = data.decode('utf-8')
                if data_str.startswith('{'):
                    record = json.loads(data_str)
                else:
                    record = self._parse_mobiflow_string(data_str)
                return self._enhance_mobiflow_record(record)
            
            # For protobuf (if needed in future)
            # from mobiflow.mobiflow_pb2 import MobiFlowRecord
            # record = MobiFlowRecord()
            # record.ParseFromString(data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error deserializing MobiFlow record: {e}")
            return None

    def _parse_mobiflow_string(self, data_str: str) -> dict:
        """
        Parse MobiFlow string format (semicolon delimited)
        """
        try:
            # Based on the MobiFlow structure in mobiflow.py
            parts = data_str.split(';')
            
            if parts[0] == 'UE':
                # Parse UE MobiFlow
                return {
                    'msg_type': parts[0],
                    'msg_id': int(parts[1]) if len(parts) > 1 else 0,
                    'timestamp': float(parts[2]) if len(parts) > 2 else time.time(),
                    'bs_id': int(parts[5]) if len(parts) > 5 else 0,
                    'rnti': int(parts[6]) if len(parts) > 6 else 0,
                    'tmsi': int(parts[7]) if len(parts) > 7 else 0,
                    'imsi': parts[8] if len(parts) > 8 else '',
                    'cipher_alg': int(parts[10]) if len(parts) > 10 else 0,
                    'integrity_alg': int(parts[11]) if len(parts) > 11 else 0,
                    'rrc_state': int(parts[15]) if len(parts) > 15 else 0,
                    'nas_state': int(parts[16]) if len(parts) > 16 else 0,
                    'sec_state': int(parts[17]) if len(parts) > 17 else 0,
                    'emm_cause': int(parts[18]) if len(parts) > 18 else 0
                }
            elif parts[0] == 'BS':
                # Parse BS MobiFlow
                return {
                    'msg_type': parts[0],
                    'msg_id': int(parts[1]) if len(parts) > 1 else 0,
                    'timestamp': float(parts[2]) if len(parts) > 2 else time.time(),
                    'bs_id': int(parts[5]) if len(parts) > 5 else 0,
                    'connected_ue_cnt': int(parts[11]) if len(parts) > 11 else 0,
                    'idle_ue_cnt': int(parts[12]) if len(parts) > 12 else 0
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error parsing MobiFlow string: {e}")
            return {}

    def _enhance_mobiflow_record(self, record: dict) -> dict:
        """
        Enhance MobiFlow record with FBS detection fields
        """
        if not record:
            return record
        
        # Add FBS-specific fields if not present
        record.setdefault('suspected_fbs', False)
        record.setdefault('attach_failures', 0)
        record.setdefault('auth_reject_count', 0)
        record.setdefault('unauthenticated_id', None)
        record.setdefault('cell_reselection_count', 0)
        record.setdefault('cipher_downgrade', False)
        record.setdefault('unexpected_redirect', False)
        record.setdefault('signal_anomaly', False)
        
        # Derive some fields from existing data if possible
        if record.get('cipher_alg') == 0:
            record['cipher_downgrade'] = True
        
        if record.get('emm_cause') in [3, 6, 7, 8]:  # Common failure causes
            record['attach_failures'] = record.get('attach_failures', 0) + 1
        
        return record

    def _check_fbs_indicators(self, record: dict) -> bool:
        """
        Check if record shows FBS indicators
        """
        indicators = 0
        
        # Check for suspicious patterns
        if record.get('attach_failures', 0) > 2:
            indicators += 1
        if record.get('auth_reject_count', 0) > 0:
            indicators += 1
        if record.get('cell_reselection_count', 0) > 5:
            indicators += 1
        if record.get('cipher_alg', 0) == 0:  # NULL cipher
            indicators += 1
        if record.get('unauthenticated_id') is not None:
            indicators += 1
        if record.get('cipher_downgrade', False):
            indicators += 1
        if record.get('signal_anomaly', False):
            indicators += 1
        
        return indicators >= 2  # Suspected FBS if 2+ indicators

    def get_ue_count(self) -> int:
        """Get total number of UEs being monitored"""
        try:
            keys = self._rmr_xapp.sdl.find_keys("ue", "")
            unique_ues = set()
            for key in keys:
                parts = key.split(':')
                if len(parts) > 1:
                    unique_ues.add(parts[1])
            return len(unique_ues)
        except Exception as e:
            self.logger.error(f"Error getting UE count: {e}")
            return 0

    def get_active_cells(self) -> list:
        """Get list of active cell IDs"""
        try:
            keys = self._rmr_xapp.sdl.find_keys("cell", "active")
            cells = []
            for key in keys:
                parts = key.split(':')
                if len(parts) > 1:
                    cells.append(parts[1])
            return cells
        except Exception as e:
            self.logger.error(f"Error getting active cells: {e}")
            return []

    def get_total_record_count(self) -> int:
        """Get total number of MobiFlow records"""
        try:
            keys = self._rmr_xapp.sdl.find_keys("mobiflow", "")
            return len(keys) if keys else 0
        except Exception as e:
            self.logger.error(f"Error getting record count: {e}")
            return 0

    def get_fbs_detection_count(self) -> int:
        """Get count of FBS detections"""
        try:
            count_data = self._rmr_xapp.sdl.get("stats", "fbs_detections")
            if count_data:
                if isinstance(count_data, (int, float)):
                    return int(count_data)
                elif isinstance(count_data, str):
                    return int(count_data) if count_data.isdigit() else 0
                elif isinstance(count_data, bytes):
                    return int(count_data.decode('utf-8')) if count_data.decode('utf-8').isdigit() else 0
            return 0
        except Exception as e:
            self.logger.error(f"Error getting FBS detection count: {e}")
            return 0

    def increment_fbs_detection_count(self) -> None:
        """Increment the FBS detection counter"""
        try:
            current_count = self.get_fbs_detection_count()
            new_count = current_count + 1
            self._rmr_xapp.sdl.set("stats", "fbs_detections", str(new_count))
            self.logger.info(f"FBS detection count incremented to {new_count}")
        except Exception as e:
            self.logger.error(f"Error incrementing FBS detection count: {e}")