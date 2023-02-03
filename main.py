
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler,StandardScaler, IndexToString, VectorIndexer, OneHotEncoder
from pyspark.ml import PipelineModel
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
import uvicorn 

class Pred(BaseModel):
    cap_shape: str
    cap_surface: str
    cap_color: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_shape: str
    stalk_root: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    stalk_color_above_ring: str
    stalk_color_below_ring: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    population: str
    habitat: str
    
def prediction(pred: Pred):
    
    schema = ['cap_shape','cap_surface','cap_color','bruises','odor','gill_attachment','gill_spacing','gill_size','gill_color','stalk_shape',
            'stalk_root','stalk_surface_above_ring','stalk_surface_below_ring','stalk_color_above_ring','stalk_color_below_ring','veil_type',
            'veil_color','ring_number','ring_type','spore_print_color','population','habitat']

    data = [tuple(j for i,j in pred.dict().items())] 
    
    spark = SparkSession.builder.appName('Mushrooms').getOrCreate()
    df = spark.createDataFrame(data=data,schema=schema)
    print(df.show())
    persistedModel = PipelineModel.load('pipelined_mushrooms')
    result = persistedModel.transform(df)
    print(result.select('probability', 'rawPrediction', 'prediction').show(10, False))
    return result.select('prediction').take(1)[0]['prediction']

app = FastAPI()

@app.post("/predict")
def predict(pred: Pred):
    print('before function')
    return prediction(pred)

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")

