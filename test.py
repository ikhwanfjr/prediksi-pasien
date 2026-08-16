from prophet.serialize import model_from_json

with open("model/prophet_model.json", "r") as fin:
    model = model_from_json(fin.read())

print(model)
print("Prophet berhasil di-load!")