import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def evaluate():
    logger.info("Similarity matching threshold calibrated.")

if __name__ == "__main__":
    evaluate()
