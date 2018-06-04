lazy val commonSettings = Seq(
    organization := "es.jarias",
    version := "0.1.0", 
    scalaVersion := "2.11.8"
)

lazy val root = (project in file(".")).settings(
    commonSettings,
    name := "spark-bnc",
 
    libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.3.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.3.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.3.0" % "provided"
)

lazy val examples = (project in file("examples")).settings(
    commonSettings,
    name := "spark-bnc-examples",
    
    libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.3.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.3.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.3.0" % "provided"

).dependsOn(root)