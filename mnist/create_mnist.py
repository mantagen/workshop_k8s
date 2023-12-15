# ./mnist/create_endpoint.py
from kubernetes import client 
from kserve import KServeClient
from kserve import constants
from kserve import utils
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1TorchServeSpec
from datetime import datetime


namespace = utils.get_default_target_namespace()
now = datetime.now()
v = now.strftime("%Y-%m-%d--%H-%M-%S")
name='mnist-{}'.format(v)
kserve_version='v1beta1'
api_version = constants.KSERVE_GROUP + '/' + kserve_version


isvc = V1beta1InferenceService(
  api_version=api_version,
  kind=constants.KSERVE_KIND,
  metadata=client.V1ObjectMeta(
    name=name,
    namespace=namespace,
    annotations={'sidecar.istio.io/inject': 'false'}
  ),
  spec=V1beta1InferenceServiceSpec(
    predictor=V1beta1PredictorSpec(
      service_account_name="sa-minio-kserve",
      pytorch=(
        V1beta1TorchServeSpec(
          storage_uri="s3://mlpipeline/models/mnist/1"
        )
      )
    )
  )
)


KServe = KServeClient()
KServe.create(isvc)