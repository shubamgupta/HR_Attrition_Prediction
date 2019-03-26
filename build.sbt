name := "HR Attrition Prediction"

version := "0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.3.2"

resolvers += Resolver.jcenterRepo

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % sparkVersion,
  "org.apache.spark" % "spark-sql_2.11" % sparkVersion,
  "org.apache.spark" % "spark-mllib_2.11" % sparkVersion,
  "com.salesforce.transmogrifai" %% "transmogrifai-core" % "0.5.0"
)