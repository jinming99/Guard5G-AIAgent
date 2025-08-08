#!/usr/bin/env python3
"""
Dataset Playback for Offline Testing
Replays PCAP and MobiFlow logs into SDL for training/testing
Location: llm_fbs_utils/dataset_playback.py
"""

import os
import json
import time
import redis
import logging
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import scapy.all as scapy


import pandas as pd
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class DatasetPlayer:
    """Replays network data into SDL for testing"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        try:
    self.redis_client.ping()
except Exception as e:
    logger.warning("Redis not reachable at init: %s (will retry on first use)", e)


        self.playback_speed = 1.0  # 1.0 = realtime, 2.0 = 2x speed
        self.loop_playback = False
        self.stop_playback = False
        
    def load_mobiflow_log(self, filepath: str) -> List[Dict]:
        """Load MobiFlow log file"""
        logger.info(f"Loading MobiFlow log: {filepath}")
        
        records = []
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                else:
                    records = [data]
        
        elif filepath.endswith('.jsonl'):
            with open(filepath, 'r') as f:
                for line in f:
                    records.append(json.loads(line))
        
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            records = df.to_dict('records')
        
        logger.info(f"Loaded {len(records)} MobiFlow records")
        return records
    
    def load_pcap(self, filepath: str) -> List[Dict]:
        """Load and parse PCAP file"""
        logger.info(f"Loading PCAP: {filepath}")
        
        packets = scapy.rdpcap(filepath)
        records = []
        
        for pkt in packets:
            record = self._parse_packet(pkt)
            if record:
                records.append(record)
        
        logger.info(f"Parsed {len(records)} packets from PCAP")
        return records
    
    def _parse_packet(self, pkt) -> Optional[Dict]:
        """Parse relevant packet into MobiFlow-like record"""
        record = {
            'timestamp': float(pkt.time),
            'packet_type': pkt.name
        }
        
        # Parse different packet types
        if pkt.haslayer('IP'):
            record['src_ip'] = pkt['IP'].src
            record['dst_ip'] = pkt['IP'].dst
        
        # Look for GTP-U packets (user plane)
        if pkt.haslayer('GTP_U_Header'):
            record['gtp_teid'] = pkt['GTP_U_Header'].teid
            record['protocol'] = 'GTP-U'
        
        # Look for NGAP packets (control plane)
        if pkt.haslayer('SCTP'):
            record['protocol'] = 'NGAP'
            record['sctp_sport'] = pkt['SCTP'].sport
            record['sctp_dport'] = pkt['SCTP'].dport
        
        # Extract RRC messages if present
        if 'RRC' in str(pkt):
            record['has_rrc'] = True
            # Further RRC parsing would go here
        
        return record if 'protocol' in record else None
    
    def inject_mobiflow_record(self, record: Dict):
        """Inject single MobiFlow record into SDL"""
        # Generate keys
        ue_id = record.get('ue_id', 'unknown')
        timestamp = record.get('timestamp', time.time())
        
        # MobiFlow record key
        key = f"mobiflow:ue:{ue_id}:{int(timestamp * 1000)}"
        
        # Store in Redis
        self.redis_client.setex(
            key,
            3600,  # 1 hour TTL
            json.dumps(record)
        )
        
        # Update latest record pointer
        self.redis_client.set(f"mobiflow:ue:{ue_id}:latest", json.dumps(record))
        
        # Update statistics
        self.redis_client.incr("stats:mobiflow_count")
        
        # Check for FBS indicators and update if detected
        if record.get('suspected_fbs'):
            self.redis_client.incr("stats:fbs_detections")
            
            # Store alert
            alert_key = f"alert:fbs:{ue_id}:{int(timestamp)}"
            alert = {
                'type': 'FBS_DETECTION',
                'ue_id': ue_id,
                'timestamp': timestamp,
                'indicators': record.get('indicators', [])
            }
            self.redis_client.setex(alert_key, 3600, json.dumps(alert))
    
    def inject_kpm_data(self, ue_id: str, kpm_data: Dict):
        """Inject KPM data into SDL"""
        key = f"kpm:{ue_id}:latest"
        self.redis_client.set(key, json.dumps(kpm_data))
        
        # Store historical KPM
        hist_key = f"kpm:{ue_id}:{int(time.time() * 1000)}"
        self.redis_client.setex(hist_key, 3600, json.dumps(kpm_data))
    
    def play_dataset(self, records: List[Dict], realtime: bool = True):
        """Play dataset records into SDL"""
        logger.info(f"Playing {len(records)} records (realtime={realtime})")
        
        if not records:
            logger.warning("No records to play")
            return
        
        # Sort by timestamp if available
        if 'timestamp' in records[0]:
            records = sorted(records, key=lambda x: x.get('timestamp', 0))
        
        # Get time range
        start_time = records[0].get('timestamp', time.time())
        last_timestamp = start_time
        
        for idx, record in enumerate(records):
            if self.stop_playback:
                logger.info("Playback stopped by user")
                break
            
            # Calculate delay for realtime playback
            if realtime and 'timestamp' in record:
                current_timestamp = record['timestamp']
                delay = (current_timestamp - last_timestamp) / self.playback_speed
                
                if delay > 0 and delay < 60:  # Cap at 1 minute
                    time.sleep(delay)
                
                last_timestamp = current_timestamp
            
            # Inject record
            self.inject_mobiflow_record(record)
            
            # Log progress
            if idx % 100 == 0:
                logger.info(f"Played {idx}/{len(records)} records")
        
        logger.info("Playback complete")
        
        if self.loop_playback and not self.stop_playback:
            logger.info("Looping playback...")
            self.play_dataset(records, realtime)
    
    def generate_synthetic_fbs_data(self, duration: int = 300) -> List[Dict]:
        """Generate synthetic FBS attack data for testing"""
        logger.info(f"Generating {duration}s of synthetic FBS data")
        
        records = []
        ue_id = "001010123456789"
        cell_id = "12345"
        fake_cell_id = "99999"
        
        current_time = time.time()
        
        # Normal operation (first 60s)
        for i in range(60):
            records.append({
                'timestamp': current_time + i,
                'ue_id': ue_id,
                'cell_id': cell_id,
                'event_type': 'MEASUREMENT_REPORT',
                'rsrp': -85 + np.random.randn() * 5,
                'rsrq': -12 + np.random.randn() * 2,
                'suspected_fbs': False,
                'attach_failures': 0,
                'auth_reject_count': 0
            })
        
        # FBS appears (60-120s)
        for i in range(60, 120):
            # Increasing signal from fake cell
            fake_rsrp = -90 + (i - 60) * 0.5
            
            records.append({
                'timestamp': current_time + i,
                'ue_id': ue_id,
                'cell_id': fake_cell_id if i > 80 else cell_id,
                'event_type': 'MEASUREMENT_REPORT',
                'rsrp': fake_rsrp,
                'rsrq': -10,
                'signal_anomaly': True,
                'suspected_fbs': i > 80
            })
            
            # Add reselection events
            if i % 10 == 0:
                records.append({
                    'timestamp': current_time + i + 0.5,
                    'ue_id': ue_id,
                    'cell_id': fake_cell_id,
                    'event_type': 'CELL_RESELECTION',
                    'cell_reselection_count': (i - 60) // 10
                })
        
        # Attack phase (120-240s)
        for i in range(120, 240):
            # Authentication failures
            if i % 20 == 0:
                records.append({
                    'timestamp': current_time + i,
                    'ue_id': ue_id,
                    'cell_id': fake_cell_id,
                    'event_type': 'AUTHENTICATION_FAILURE',
                    'auth_reject_count': (i - 120) // 20,
                    'suspected_fbs': True
                })
            
            # Attach failures
            if i % 30 == 0:
                records.append({
                    'timestamp': current_time + i,
                    'ue_id': ue_id,
                    'cell_id': fake_cell_id,
                    'event_type': 'RRC_CONNECTION_REJECT',
                    'attach_failures': (i - 120) // 30,
                    'suspected_fbs': True
                })
            
            # Cipher downgrade
            if i == 150:
                records.append({
                    'timestamp': current_time + i,
                    'ue_id': ue_id,
                    'cell_id': fake_cell_id,
                    'event_type': 'SECURITY_MODE_COMMAND',
                    'cipher_algorithm': 'NULL',
                    'cipher_downgrade': True,
                    'suspected_fbs': True
                })
        
        # Recovery phase (240-300s)
        for i in range(240, duration):
            records.append({
                'timestamp': current_time + i,
                'ue_id': ue_id,
                'cell_id': cell_id,  # Back to legitimate cell
                'event_type': 'MEASUREMENT_REPORT',
                'rsrp': -85 + np.random.randn() * 5,
                'rsrq': -12 + np.random.randn() * 2,
                'suspected_fbs': False,
                'recovery_phase': True
            })
        
        logger.info(f"Generated {len(records)} synthetic records")
        return records
    
    def clear_sdl(self):
        """Clear all test data from SDL"""
        logger.warning("Clearing SDL test data...")
        
        patterns = [
            "mobiflow:*",
            "kpm:*",
            "alert:*",
            "stats:*"
        ]
        
        for pattern in patterns:
            batch = []
            for k in self.redis_client.scan_iter(match=pattern, count=1000):
                batch.append(k)
                if len(batch) >= 1000:
                    self.redis_client.delete(*batch)
                    batch.clear()
            if batch:
                self.redis_client.delete(*batch)

    
    def get_playback_stats(self) -> Dict:
        """Get playback statistics from SDL"""
        stats = {
            'mobiflow_count': self.redis_client.get("stats:mobiflow_count") or 0,
            'fbs_detections': self.redis_client.get("stats:fbs_detections") or 0,
            'unique_ues': len(self.redis_client.keys("mobiflow:ue:*:latest")),
            'alerts': len(self.redis_client.keys("alert:*"))
        }
        return stats

# ============================================================================
# Dataset Conversion Utilities
# ============================================================================

class DatasetConverter:
    """Convert between different dataset formats"""
    
    @staticmethod
    def pcap_to_mobiflow(pcap_file: str, output_file: str):
        """Convert PCAP to MobiFlow format"""
        player = DatasetPlayer()
        records = player.load_pcap(pcap_file)
        
        # Enhance records with MobiFlow fields
        for record in records:
            record['ue_id'] = record.get('gtp_teid', 'unknown')
            record['cell_id'] = 'cell_1'  # Default
            record['event_type'] = 'PACKET_DATA'
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(records, f, indent=2)
        
        logger.info(f"Converted {len(records)} records to {output_file}")
    
    @staticmethod
    def csv_to_mobiflow(csv_file: str, output_file: str):
        """Convert CSV to MobiFlow format"""
        df = pd.read_csv(csv_file)
        
        # Map CSV columns to MobiFlow fields
        records = []
        for _, row in df.iterrows():
            record = {
                'timestamp': row.get('timestamp', time.time()),
                'ue_id': str(row.get('ue_id', 'unknown')),
                'cell_id': str(row.get('cell_id', 'unknown')),
                'event_type': row.get('event_type', 'UNKNOWN'),
                # Add more field mappings as needed
            }
            records.append(record)
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(records, f, indent=2)
        
        logger.info(f"Converted {len(records)} records to {output_file}")

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dataset Playback Tool')
    
    parser.add_argument('command', choices=['play', 'generate', 'clear', 'stats', 'convert'])
    parser.add_argument('--file', help='Input file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['mobiflow', 'pcap', 'csv'], default='mobiflow')
    parser.add_argument('--realtime', action='store_true', help='Realtime playback')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed')
    parser.add_argument('--loop', action='store_true', help='Loop playback')
    parser.add_argument('--duration', type=int, default=300, help='Duration for synthetic data')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    # Initialize player
    player = DatasetPlayer(args.redis_host, args.redis_port)
    player.playback_speed = args.speed
    player.loop_playback = args.loop
    
    if args.command == 'play':
        if not args.file:
            print("Please specify input file with --file")
            return
        
        # Load dataset
        if args.format == 'mobiflow':
            records = player.load_mobiflow_log(args.file)
        elif args.format == 'pcap':
            records = player.load_pcap(args.file)
        else:
            print(f"Unsupported format: {args.format}")
            return
        
        # Play dataset
        player.play_dataset(records, realtime=args.realtime)
    
    elif args.command == 'generate':
        # Generate synthetic data
        import numpy as np  # Import here to avoid dependency if not used
        
        records = player.generate_synthetic_fbs_data(args.duration)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(records, f, indent=2)
            print(f"Saved {len(records)} records to {args.output}")
        else:
            # Play directly
            player.play_dataset(records, realtime=args.realtime)
    
    elif args.command == 'clear':
        player.clear_sdl()
        print("SDL cleared")
    
    elif args.command == 'stats':
        stats = player.get_playback_stats()
        print("Playback Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.command == 'convert':
        if not args.file or not args.output:
            print("Please specify input and output files")
            return
        
        converter = DatasetConverter()
        
        if args.format == 'pcap':
            converter.pcap_to_mobiflow(args.file, args.output)
        elif args.format == 'csv':
            converter.csv_to_mobiflow(args.file, args.output)
        else:
            print(f"Conversion from {args.format} not supported")

if __name__ == "__main__":
    main()