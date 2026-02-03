from sec_edgar_downloader import Downloader
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the downloader
dl = Downloader("rag-assistant", os.getenv("SEC_EMAIL"), "./data")

# List of tickers we want to study
tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

def fetch_reports():
    # Download the most recent 10-K for each company
    for ticker in tickers:
        print(f"Downloading latest 10-K for {ticker}...")
        dl.get("10-K", ticker, limit=1, download_details=True)

if __name__ == "__main__":
    fetch_reports()
    print("\nDownload complete!")