import os
import prefect
from prefect import task, Flow, Parameter
from prefect.run_configs import LocalRun

@task
def hello_task(name):
    greeting = os.environ.get("GREETING")
    logger = prefect.context.get("logger")
    logger.info(f"{greeting}, {name}")

# flow = Flow("hello-flow", tasks=[hello_task])
with Flow("hello-flow") as flow:
    people = Parameter("people", default=["Arthur", "Ford", "Marvin"])

    hello_task.map(people)

# flow.run() This command is used for running locally
flow.run_config = LocalRun(env={"GREETING": "Hello"})

flow.register(project_name="operationalize-ml-microservice")
flow.run_agent()