import breeze.features.FeatureVector
import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.readers._
import org.apache.spark.sql.SparkSession
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry.{OpLogisticRegression, OpRandomForestClassifier}
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.tuning.{DataBalancer, DataCutter, DataSplitter}
import org.apache.spark.ml.tuning.ParamGridBuilder

case class HRStruct
(
  id: Option[String],
  age: Option[Int],
  gender: Option[String],
  educationBack: Option[String],
  maritalStatus: Option[String],
  empDept: Option[String],
  employRole: Option[String],
  businessTravel: Option[String],
  distanceFromHome: Option[Double],
  empEduLvl: Option[Int],
  empEnvSat: Option[Int],
  empHourRate: Option[Double],
  empJobInv: Option[Int],
  empJobLvl: Option[Int],
  empJobSat: Option[Int],
  numCompaniesWorked: Option[Int],
  overtime: Option[String],
  empLastSal: Option[Int],
  empRel: Option[Int],
  totalWorkExp: Option[Int],
  trainingTime: Option[Int],
  empWorkLife: Option[Int],
  experienceYrCompany: Option[Int],
  experienceYrRole: Option[Int],
  yrsSinceLastPromo: Option[Int],
  yrsWithManager: Option[Int],
  attrition: Double,
  rating: Option[Int]
)
object HRMain {
  def main(args: Array[String]): Unit = {
    implicit val spark = SparkSession.builder.config("spark.master", "local").getOrCreate()
    import spark.implicits._
    val csvFilePath = "../HR Attrition Prediction/src/main/resources/HR_Dataset.csv"

    // Read Titanic data as a DataFrame
    val pathToData = Option(csvFilePath)
    val passengersData = DataReaders.Simple.csvCase[HRStruct](pathToData, key = _.id.toString).readDataset().toDF()

    // Automated feature engineering
    val (attrition, features) = FeatureBuilder.fromDataFrame[RealNN](passengersData, response = "attrition")
    val featureVector = features.transmogrify()

//    // Automated feature selection
    val checkedFeatures = attrition.sanityCheck(featureVector, checkSample = 1.0, removeBadFeatures = true)

    println("Feature Vector")
    println(features)

    val randomSeed = 112233L

    // Automated model selection
    val lr = new OpLogisticRegression()
    val rf = new OpRandomForestClassifier()
    val gb = new OpGBTClassifier()
    val models = Seq(
      lr -> new ParamGridBuilder()
        .addGrid(lr.regParam, Array(0.05, 0.1))
        .addGrid(lr.elasticNetParam, Array(0.01))
        .build(),
      rf -> new ParamGridBuilder()
        .addGrid(rf.maxDepth, Array(5, 10))
        .addGrid(rf.minInstancesPerNode, Array(10, 20, 30))
        .addGrid(rf.seed, Array(randomSeed))
        .build(),
      gb -> new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(5, 10))
      .addGrid(rf.minInstancesPerNode, Array(10, 20, 30))
      .addGrid(rf.seed, Array(randomSeed))
      .build()
    )
    val splitter = DataSplitter(seed = randomSeed, reserveTestFraction = 0.1)


    val prediction = BinaryClassificationModelSelector
      .withTrainValidationSplit(Option(DataBalancer(reserveTestFraction = 0.1, sampleFraction = 0.2,seed=randomSeed)),modelsAndParameters = models)
      .setInput(attrition, checkedFeatures)
      .getOutput()




//    // Automated model selection
//    val prediction = BinaryClassificationModelSelector()
//      .setInput(attrition, checkedFeatures).getOutput()

    val model = new OpWorkflow().setInputDataset(passengersData).setResultFeatures(prediction).train()

    println("Model summary:\n" + model.summaryPretty())


//    val saveWorkflowPath = "../HR Attrition Prediction/src/main/resources/model-v0.1"
//    model.save(saveWorkflowPath)
  }
}

