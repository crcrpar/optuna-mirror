import optuna

server = optuna.rpc.RpcServer()
server.start()
print("# Server Started")

client = optuna.rpc.RpcClient()
print("# Client connected")

session = client.create_session()
print("# Session created: {}".format(session))

study = optuna.rpc.create_study(session)
print("# Study created: {}".format(study))

print("# Trials: {}".format(study.trials()))

print("# Set Attr: 'hello' => 10")
study.set_user_attr('hello', 10)

print("# Study Summary: {}".format(study.study_summary()))
