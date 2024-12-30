using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using Microsoft.ML;

namespace ReviewRatingApp
{
    public partial class MainWindow : Window
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private readonly List<ReviewData> _correctedData;
        private readonly string _modelPath;

        public MainWindow()
        {
            InitializeComponent();

            // Initialize MLContext and corrected data
            _mlContext = new MLContext();
            _correctedData = new List<ReviewData>();
            _modelPath = @"model1.zip";

            // Load the model
          
                DataViewSchema modelSchema;
                _model = _mlContext.Model.Load(_modelPath, out modelSchema);
                MessageBox.Show("Model loaded successfully!", "Success");
            
          
        }

        private void PredictButton_Click(object sender, RoutedEventArgs e)
        {
            string reviewText = ReviewTextBox.Text;

            if (string.IsNullOrWhiteSpace(reviewText))
            {
                MessageBox.Show("Please enter a review.", "Error");
                return;
            }

            if (_model == null)
            {
                MessageBox.Show("Model is not loaded. Please ensure the model is available.", "Error");
                return;
            }

            try
            {
                var predictionEngine = _mlContext.Model.CreatePredictionEngine<ReviewData, ReviewPrediction>(_model);
                var reviewData = new ReviewData { Review = reviewText };
                var prediction = predictionEngine.Predict(reviewData);

                PredictedRatingLabel.Content = $"Predicted Rating: {prediction.Rating:F2}";
                CorrectedRatingTextBox.Text = prediction.Rating.ToString("F2");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during prediction: {ex.Message}", "Error");
            }
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            string reviewText = ReviewTextBox.Text;

            if (string.IsNullOrWhiteSpace(reviewText))
            {
                MessageBox.Show("Please enter a review.", "Error");
                return;
            }

            if (float.TryParse(CorrectedRatingTextBox.Text, out float correctedRating))
            {
                _correctedData.Add(new ReviewData { Review = reviewText, Rating = correctedRating });
                MessageBox.Show("Corrected rating saved. Retrain the model to apply the changes.", "Saved");
            }
            else
            {
                MessageBox.Show("Invalid corrected rating. Please enter a valid number.", "Error");
            }
        }

        private void RetrainButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Load original data
                string originalDataPath = @"tripadvisor_hotel_clean.csv";
                if (!File.Exists(originalDataPath))
                {
                    MessageBox.Show($"Original data file not found: {originalDataPath}", "Error");
                    return;
                }

                IDataView originalData = _mlContext.Data.LoadFromTextFile<ReviewData>(
                    originalDataPath,
                    hasHeader: true,
                    separatorChar: ',');

                // Combine original and corrected data
                var originalDataList = _mlContext.Data.CreateEnumerable<ReviewData>(originalData, reuseRowObject: false).ToList();
                var combinedList = originalDataList.Concat(_correctedData).ToList();
                IDataView combinedData = _mlContext.Data.LoadFromEnumerable(combinedList);

                // Define the pipeline
                var pipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(ReviewData.Review))
                    .Append(_mlContext.Transforms.CopyColumns("Label", nameof(ReviewData.Rating)))
                    .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));

                // Retrain the model
                _model = pipeline.Fit(combinedData);

                // Save the updated model
                _mlContext.Model.Save(_model, combinedData.Schema, _modelPath);

                MessageBox.Show("Model retrained and saved successfully!", "Success");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during model retraining: {ex.Message}", "Error");
            }
        }

        private void CorrectedRatingTextBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
        {

        }
    }
}