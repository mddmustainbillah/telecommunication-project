from telecommunication.pipeline.pipeline import main_pipeline

if __name__ == "__main__":
    main_pipeline.serve(
        name="telco_commission_prediction",
        version="1",
        tags=["telco"],
        # schedule=None
    )
