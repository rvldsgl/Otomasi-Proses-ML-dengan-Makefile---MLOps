from joblib import load

def deploy_model():

    model = load('models/model.joblib')

    print("Model ready for deployment")

if __name__ == "__main__":
    deploy_model()
