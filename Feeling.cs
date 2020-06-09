using System;
using Microsoft.ML.Data;

namespace feelings
{
    public class Feeling
    {
        public class DataFeeling
        {
            [LoadColumn(0)]
            public string Text;
            [LoadColumn(1), ColumnName("Label")]
            public bool Label;
            
        }

        public class PredictFeeling : DataFeeling
        {
            [ColumnName("PredictedLabel")]
            public bool Prediction { get; set; }

            public float Probability {get;set;}
            public float Score {get;set;}
            
            
        }
        
    }
}