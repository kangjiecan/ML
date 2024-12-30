using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReviewRatingApp
{
    public class ReviewPrediction
    {
        [ColumnName("Score")]
        public float Rating { get; set; }
    }
}
