from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, Response

app = FastAPI()


@app.get("/chart/{timeframe}")
def get_btc_chart(timeframe: str):
    file_map = {
        "15T": "BTCUSD_M1_20250604_to_20250620_bars_15T_enriched.csv",
        "1H": "BTCUSD_M1_20250604_to_20250620_bars_1H_enriched.csv",
        "4H": "BTCUSD_M1_20250604_to_20250620_bars_4H_enriched.csv",
        "1D": "BTCUSD_M1_20250604_to_20250620_bars_1D_enriched.csv",
    }

    if timeframe not in file_map:
        return {"error": "Unsupported timeframe."}

    df = pd.read_csv(f"./BAR_DATA/BTCUSD/BTCUSD_bars/{file_map[timeframe]}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['close'], label='Close')
    plt.title(f"BTCUSD Close Price - {timeframe}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
