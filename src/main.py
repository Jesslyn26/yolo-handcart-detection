from data_pipeline.bronze_layer import run_bronze
from data_pipeline.silver_layer import run_silver
from data_pipeline.gold_layer import run_gold
from ml_pipeline.train import train_yolo
from ml_pipeline.evaluate import evaluate_model
import datetime

def main():

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Running Bronze...")
    bronze_root = run_bronze(batch_id = batch_id)

    print("Running Silver...")
    silver_root = run_silver(bronze_root, batch_id=batch_id)

    print("Running Gold...")
    gold_results = run_gold(silver_root, batch_id = batch_id)

    print("Training YOLO...")
    model = train_yolo(batch_id = batch_id, data_yaml = gold_results["data_yaml"])

    print("Evaluating model...")
    evaluate_model(model)

if __name__ == "__main__":
    main()