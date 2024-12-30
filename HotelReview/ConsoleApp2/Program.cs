using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace HotelReviewAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create ML.NET context
            var mlContext = new MLContext();

            Console.WriteLine($"Current Directory: {Environment.CurrentDirectory}");
            Console.WriteLine($"Base Directory: {AppDomain.CurrentDomain.BaseDirectory}");

            // Load data
            string dataPath ="tripadvisor_hotel_clean.csv";
            var data = mlContext.Data.LoadFromTextFile<ReviewData>(dataPath, hasHeader: true, separatorChar: ',');

            // Debug: Print the first 10 rows of loaded data
            Console.WriteLine("First 10 rows of loaded data:");
            var rawData = mlContext.Data.CreateEnumerable<ReviewData>(data, reuseRowObject: false);
            foreach (var row in rawData.Take(10))
            {
                Console.WriteLine($"Review: {row.Review}, Rating: {row.Rating}");
            }

            // Split data into train and test sets
            var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            // Define the pipeline
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(ReviewData.Review))
                .Append(mlContext.Transforms.CopyColumns("Label", nameof(ReviewData.Rating)))
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));

            // Train the model
            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Evaluate the model
            Console.WriteLine("Evaluating the model...");
            var predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            // Output evaluation metrics
            Console.WriteLine($"R^2: {metrics.RSquared:F2}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F2}");

            // Save the model
            string modelPath = "model1.zip";
            mlContext.Model.Save(model, trainData.Schema, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");

            // Load the saved model
            Console.WriteLine("Loading the saved model...");
            DataViewSchema modelSchema;
            ITransformer loadedModel = mlContext.Model.Load(modelPath, out modelSchema);
            Console.WriteLine("Model loaded successfully!");

            // Test the loaded model with a sample review
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ReviewData, ReviewPrediction>(loadedModel);
            var sampleReview = new ReviewData { Review = "The hotel was clean, the bed was dirty." };
            var prediction = predictionEngine.Predict(sampleReview);

            Console.WriteLine($"Predicted Rating for sample review: {prediction.Rating:F2}");
        }
    }
}
