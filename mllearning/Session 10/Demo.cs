using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;

namespace mllearning.Session_10
{
     class Demo
    {

       public static void Execute()
        {
            MLContext mlContext = new MLContext();

            List<InputModel> data = new List<InputModel>()
            {

                new InputModel{ YearOfExperiences =1.3f , Salary = 46500},
                 new InputModel{ YearOfExperiences =2.1f , Salary = 73899},
                  new InputModel{ YearOfExperiences =3.2f , Salary = 272782},
                   new InputModel{ YearOfExperiences =9.1f , Salary = 522282},
                    new InputModel{ YearOfExperiences =4.3f , Salary = 537728},
                     new InputModel{ YearOfExperiences =5.1f , Salary = 68382},
                      new InputModel{ YearOfExperiences =6.3f , Salary = 37382},
                       new InputModel{ YearOfExperiences =3.9f , Salary = 6282829},
                        new InputModel{ YearOfExperiences =6.7f , Salary = 638282},
                         new InputModel{ YearOfExperiences =54.5f , Salary = 638282},
                          new InputModel{ YearOfExperiences =5.8f , Salary = 73838}


            };

            IDataView TrainingData = mlContext.Data.LoadFromEnumerable(data);

            var Estimator = mlContext.Transforms.Concatenate("Features", new[] { "YearOfExperience" });

            var Pipeline = Estimator.Append(mlContext.Regression.Trainers.Sdca(labelColumnName:"Salary", maximumNumberOfIterations:100));

            var model = Pipeline.Fit(TrainingData);

            var PredictEngine = mlContext.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            var Experience = new InputModel { YearOfExperiences =5 };

            var result = PredictEngine.Predict(Experience);

            Console.WriteLine($"Approx salary for {Experience.YearOfExperiences} Year of Experience Will Be : {result.Salary}");

        }
    }
}
