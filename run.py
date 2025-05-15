from app import app
import os

if __name__ == "__main__":
    # Set port explicitly to 8080 for Railway
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)