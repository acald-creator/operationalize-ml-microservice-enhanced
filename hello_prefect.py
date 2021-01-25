import prefect
from prefect import task, Flow

@task
def hello_task():
    logger = prefect.context.get("logger")
    logger.info("Hello world!")

flow = Flow("hello-flow", tasks=[hello_task])

# flow.run() This command is used for running locally

flow.register(project_name="operationalize-ml-microservice")
flow.run_agent()