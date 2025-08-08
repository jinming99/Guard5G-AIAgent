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

from .hwxapp import HWXapp
import threading
import logging
import os

# LLM API Support - Check if module exists before importing
try:
    from . import llm_control_api
    LLM_API_AVAILABLE = True
except ImportError:
    LLM_API_AVAILABLE = False
    logging.warning("llm_control_api module not found - LLM API will not be available")


def initialize_llm_api(sdl_manager):
    """Initialize and start the LLM Control API"""
    if not LLM_API_AVAILABLE:
        logging.warning("LLM Control API module not available")
        return None
    
    try:
        # Pass the SDL manager instance to the API
        llm_control_api.set_sdl_manager(sdl_manager)
        
        # Start the API server in a background thread
        api_thread = llm_control_api.start_in_thread()
        
        print("LLM Control API started on port 8090")
        logging.info("LLM Control API initialized successfully")
        return api_thread
    except Exception as e:
        logging.error(f"Failed to start LLM Control API: {e}")
        return None


def launchXapp():
    hwxapp = HWXapp()
    
    # Initialize LLM API if enabled
    enable_llm_api = os.getenv('ENABLE_LLM_API', 'true').lower() == 'true'
    
    if enable_llm_api and LLM_API_AVAILABLE:
        try:
            # Get SDL manager from the xApp
            # Note: This assumes hwxapp has an sdl_manager attribute
            # You may need to adjust based on actual implementation
            if hasattr(hwxapp, 'sdl_manager'):
                llm_api_thread = initialize_llm_api(hwxapp.sdl_manager)
            else:
                logging.warning("SDL manager not found in HWXapp - LLM API not started")
        except Exception as e:
            logging.error(f"Failed to initialize LLM Control API: {e}")
    
    # Start the main xApp
    hwxapp.start()


if __name__ == "__main__":
    launchXapp()