import os
import boto3
import sagemaker
import sagemaker.session
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep, CacheConfig, TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)

# How many instances to use when processing
processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=1
)

# What instance type to use for processing
processing_instance_type = ParameterString(
    name="ProcessingInstanceType",
    default_value="ml.m5.large"
)

# What instance type to use for training
training_instance_type = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.m5.xlarge"
)

# Where the input data is stored
input_data = ParameterString(
    name="InputData",
    default_value='s3://sagemaker-eu-west-1-641758013508/sagemaker/credit-xgboost/data/UCI_Credit_Card.csv',
)

# What is the default status of the model when registering with model registry.
model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"
)

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Create cache configuration
    cache_config = CacheConfig(enable_caching=True, expire_after="T30m")

    # Create SKlean processor object
    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name="credit-processing-job"
    )

    # Use the sklearn_processor in a Sagemaker pipelines ProcessingStep
    step_preprocess_data = ProcessingStep(
        name="PreprocessCreditData",
        processor=sklearn_processor,
        cache_config=cache_config,
        inputs=[
          ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),  
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test"),
            ProcessingOutput(output_name="baseline_with_headers", source="/opt/ml/processing/output/baseline")
        ],
        code=os.path.join(BASE_DIR, "preprocessing.py"),
    )


    # Where to store the trained model
    model_path = f"s3://{default_bucket}/CreditTrain"

    # Fetch container to use for training
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.2-2",
        py_version="py3",
        instance_type=training_instance_type,
    )

    # Create XGBoost estimator object
    xgb_estimator = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        role=role,
        disable_profiler=True,
    )

    # Specify hyperparameters
    xgb_estimator.set_hyperparameters(max_depth=5,
                            eta=0.2,
                            gamma=4,
                            min_child_weight=6,
                            subsample=0.8,
                            objective='binary:logistic',
                            num_round=25)

    # Use the xgb_estimator in a Sagemaker pipelines ProcessingStep. 
    # NOTE how the input to the training job directly references the output of the previous step.
    step_train_model = TrainingStep(
        name="TrainCreditModel",
        estimator=xgb_estimator,
        cache_config=cache_config,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv"
            )
        },
    )

    # Create ScriptProcessor object.
    evaluate_model_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="script-credit-eval",
        role=role,
    )

    # Create a PropertyFile
    # We use a PropertyFile to be able to reference outputs from a processing step, for instance to use in a condition step, which we'll see later on.
    # For more information, visit https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    # Use the evaluate_model_processor in a Sagemaker pipelines ProcessingStep. 
    step_evaluate_model = ProcessingStep(
        name="EvaluateCreditModel",
        processor=evaluate_model_processor,
        cache_config=cache_config,
        inputs=[
            ProcessingInput(
                source=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluation.py"),
        property_files=[evaluation_report],
    )


    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_evaluate_model.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )

    # Crete a RegisterModel step, which registers your model with Sagemaker Model Registry.
    step_register_model = RegisterModel(
        name="RegisterCreditModel",
        estimator=xgb_estimator,
        model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge", "ml.m5.large"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )


    # Create Processor object using the model monitor image
    baseline_processor = sagemaker.processing.Processor(
        base_job_name="credit-risk-baseline-processor",
        image_uri=sagemaker.image_uris.retrieve(framework='model-monitor', region='eu-west-1'),
        role=role,
        instance_count=1,
        instance_type=processing_instance_type,
        env = {
            "dataset_format": "{\"csv\": {\"header\": true} }",
            "dataset_source": "/opt/ml/processing/sm_input",
            "output_path": "/opt/ml/processing/sm_output",
            "publish_cloudwatch_metrics": "Disabled"
        }
    )

    # Create a Sagemaker Pipeline step, using the baseline_processor.
    step_create_data_baseline = ProcessingStep(
        name="CreateModelQualityBaseline",
        processor=baseline_processor,
        cache_config=cache_config,
        inputs=[
            ProcessingInput(
                source=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                    "baseline_with_headers"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/sm_input",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/sm_output",
                destination="s3://{}/{}/baseline".format(default_bucket, base_job_prefix),
                output_name="baseline_result",
            )
        ],
    )



    # Create Condition
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=step_evaluate_model,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value"
        ),
        right=0.7
    )

    # Create a Sagemaker Pipelines ConditionStep, using the condition we just created.
    step_cond = ConditionStep(
        name="AccuracyCondition",
        conditions=[cond_gte],
        if_steps=[step_register_model],
        else_steps=[], 
    )

    from sagemaker.workflow.pipeline import Pipeline

    # Create a Sagemaker Pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type, 
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_preprocess_data, step_train_model, step_evaluate_model, step_create_data_baseline, step_cond],
    )
    
    return pipeline