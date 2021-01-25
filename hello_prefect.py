import prefect
from prefect import task, Flow, Parameter

@task
def hello_task(name):
    logger = prefect.context.get("logger")
    logger.info(f"Hello, {name}")

# flow = Flow("hello-flow", tasks=[hello_task])
with Flow("hello-flow") as flow:
    people = Parameter("people", default=["Arthur", "Ford", "Marvin"])

    hello_task.map(people)

# flow.run() This command is used for running locally

flow.register(project_name="operationalize-ml-microservice")
flow.run_agent()