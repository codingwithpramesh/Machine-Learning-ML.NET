﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mllearning.Session_10
{
     class ResultModel
    {
        [ColumnName("Score")]
        public float Salary { get; set; }
    }
}
