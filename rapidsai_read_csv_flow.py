from prefect import task, Flow, Parameter
from prefect.executors import LocalDaskExecutor
import cudf

@task()
def read_data_file(file_path):
    df = cudf.read_csv(file_path)
    return df

with Flow("Read Data") as cudf_read_data_flow:
    file_list = Parameter("file_list")
    dfs = read_data_file.map(file_list)

executor = LocalDaskExecutor(
    cluster_class="dask_cuda.LocalCUDACluster"
)

cudf_read_data_flow.run(
    file_list = ["./model_data/housing.csv"],
    executor=executor
)

cudf_read_data_flow.register(project_name="operationalize-ml-microservice")
cudf_read_data_flow.run_agent()