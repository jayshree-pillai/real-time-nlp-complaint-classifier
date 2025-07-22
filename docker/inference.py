from sagemaker.model import Model
from sagemaker import Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

role = "<your-sagemaker-execution-role>"
ecr_image = "<your_ecr_image_uri>"  # after docker push
model_data = "s3://complaint-classifier-jp2025/customHeadBert/model.tar.gz"

sm_model = Model(
    image_uri=ecr_image,
    model_data=model_data,
    role=role,
    predictor_cls=Predictor,
    sagemaker_session=Session()
)

predictor = sm_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)
