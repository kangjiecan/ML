using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class ReviewData
{
    [LoadColumn(1)]
    public string Review { get; set; }

    [LoadColumn(0)]
    public float Rating { get; set; }
}
