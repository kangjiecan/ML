using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReviewRatingApp
{
    public class ReviewData
    {
        [LoadColumn(1)] // Adjust this index based on your CSV structure
        public string Review { get; set; }

        [LoadColumn(0)] // Adjust this index based on your CSV structure
        public float Rating { get; set; }
    }
}
