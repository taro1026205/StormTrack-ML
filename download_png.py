import os
import requests
import argparse
from datetime import datetime, timedelta

def download_radar_images(start_time, end_time, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    print(f"Starting to download images from {start_time} to {end_time}...")
    
    current_time = start_time
    while current_time <= end_time:
        yy = current_time.strftime("%Y")
        mm = current_time.strftime("%m")
        dd = current_time.strftime("%d")
        hour_minute = current_time.strftime("%H_%M")
        
        base_url = f"https://kttvnb.info/kttvnb-admin/public/products/RADAR_NEW/{yy}/{mm}/{dd}"
        file_name = f"{hour_minute}.png"
        url = f"{base_url}/{file_name}"
        
        file_path = os.path.join(save_folder, file_name)
        
        if not os.path.exists(file_path):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Successfully downloaded: {file_name}")
                else:
                    print(f"❌ Image not found on server: {file_name}")
            except Exception as e:
                print(f"⚠️ Connection error while downloading {file_name}: {e}")
        else:
            print(f"⏭️ File already exists, skipping: {file_name}")
            
        current_time += timedelta(minutes=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Radar PNGs")
    parser.add_argument("--start", type=str, required=True, help="Start time (YYYY-MM-DD HH:MM)")
    parser.add_argument("--end", type=str, required=True, help="End time (YYYY-MM-DD HH:MM)")
    parser.add_argument("--folder", type=str, default="file/huge_storm_data/png", help="Output folder")
    args = parser.parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
    
    download_radar_images(start, end, args.folder)
