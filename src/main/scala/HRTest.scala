import breeze.features.FeatureVector
import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.readers._
import org.apache.spark.sql.SparkSession
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry.{OpLogisticRegression, OpRandomForestClassifier}
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.tuning.DataCutter

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
object HRTest {
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

    // Automated model selection
    val prediction = BinaryClassificationModelSelector()
      .setInput(attrition, checkedFeatures).getOutput()

    val workflow = new OpWorkflow().setInputDataset(passengersData).setResultFeatures(prediction)

    val workflowModel: OpWorkflowModel = workflow.loadModel(path = "../HR Attrition Prediction/src/main/resources/model-v0.1")


    val df = workflowModel.setInputDataset(passengersData).computeDataUpTo(checkedFeatures)


    println(df.show(truncate = false))

  }
}

