
import os
import time
import json
import hmac
import base64
import hashlib
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_CODE = os.getenv("DX_ACCOUNT_CODE")
PUBLIC_KEY = os.getenv("DX_PUBLIC_KEY")
PRIVATE_KEY = os.getenv("DX_PRIVATE_KEY").encode()
DX_BASE_URL = "https://demo.dx.trade/dxsca-web"

def generate_hmac_signature(method, uri, content):
    timestamp = str(int(time.time() * 1000))
    hash_candidate = f"Method={method}\nContent={content}\nURI={uri}\nTimestamp={timestamp}"
    signature = hmac.new(PRIVATE_KEY, hash_candidate.encode("utf-8"), hashlib.sha256).digest()
    hmac_hash = base64.b64encode(signature).decode()
    auth_header = f'DXAPI principal="{PUBLIC_KEY}",timestamp={timestamp},hash="{hmac_hash}"'
    return auth_header

def place_order(symbol, price, quantity=0.01):
    uri = f"/accounts/{ACCOUNT_CODE}/orders"
    content = json.dumps({
        "clientOrderId": f"zdx-entry-{int(time.time())}",
        "type": "LIMIT",
        "instrument": symbol,
        "quantity": quantity,
        "side": "BUY",
        "limitPrice": price,
        "tif": "GTC"
    })
    auth_header = generate_hmac_signature("POST", uri, content)
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(
            DX_BASE_URL + uri, headers=headers, data=content, timeout=10
        )
        logging.info("[ZDX] Order Status: %s %s", response.status_code, response.text)
        return response
    except requests.RequestException as e:
        logging.error("[ZDX] Order placement failed: %s", e)
        return None
