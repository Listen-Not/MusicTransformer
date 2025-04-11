import os
import re
import requests
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

# 多个目标网站，仅需填写 base_url
TARGET_SITES = [
    "http://piano-midi.de/midicoll.htm",
    # "https://www.kunstderfuge.com/",这个要付费
    # 添加更多入口页面：
    # "https://example.com/index.htm",
]

# 保存路径 & 日志路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(BASE_DIR, "Input")
log_path = os.path.join(BASE_DIR, "failed_downloads.log")
os.makedirs(save_dir, exist_ok=True)

# 清空旧日志
with open(log_path, "w") as log_file:
    log_file.write("Failed MIDI Downloads:\n")

# ---------------- 工具函数 ---------------- #


def get_download_base(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def fetch(url):
    try:
        response = requests.get(url, timeout=(20, 30))
        response.raise_for_status()
        return response.text
    except Exception as e:
        tqdm.write(f"❌ Failed to fetch {url}: {e}")
        return None


def extract_subpages(html, base):
    return list(set([urljoin(base, link) for link in re.findall(r'href="([^"]+\.htm[l]?)"', html, re.IGNORECASE)]))


def extract_midi_links(html, base):
    return list(set([urljoin(base, link) for link in re.findall(r'href="([^"]+\.(?:mid|midi))"', html, re.IGNORECASE)]))


def download_file(url, save_dir, log_path):
    file_name = os.path.basename(urlparse(url).path)
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        return

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"{url}  # {e}\n")


# ---------------- 主流程 ---------------- #

for base_url in TARGET_SITES:
    download_base = get_download_base(base_url)

    print(f"\n🌐 Fetching base page: {base_url}")
    main_html = fetch(base_url)
    if not main_html:
        continue

    # 提取子页面
    subpages = extract_subpages(main_html, download_base)
    print(f"🔗 Found {len(subpages)} subpages from {base_url}")

    # 提取所有 MIDI 链接
    all_midi_links = []
    for subpage in tqdm(subpages, desc=f"🔍 Scanning subpages of {urlparse(base_url).netloc}", leave=False):
        html = fetch(subpage)
        if html:
            midi_links = extract_midi_links(html, subpage)
            all_midi_links.extend(midi_links)

    # 去重
    all_midi_links = list(set(all_midi_links))
    print(f"🎼 Found {len(all_midi_links)} MIDI files in {base_url}")

    # 下载
    for url in tqdm(
        all_midi_links, desc=f"🎹 Downloading from {urlparse(base_url).netloc}", unit="file", unit_scale=True
    ):
        download_file(url, save_dir, log_path)

print("\n✅ All downloads complete.")
print(f"🎵 MIDI files saved to: {save_dir}")
print(f"📄 Failed downloads logged in: {log_path}")
