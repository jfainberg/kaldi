// featbin/feat-to-ali.cc

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Reads an archive of features and writes alignments\n"
        "that maps utterance frames to a generic value.\n"
        "Usage: feat-to-ali [options] <in-rspecifier> <out-wspecifier>\n"
        "e.g.: feat-to-ali scp:feats.scp ark,t:ali\n";
    
    ParseOptions po(usage);

    int32 value = 0;
    po.Register("value", &value, "Integer to initialise all assignments to.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    if (po.NumArgs() == 2) {
      std::string rspecifier = po.GetArg(1);
      std::string wspecifier = po.GetArg(2);

      Int32VectorWriter ali_writer(wspecifier);
      SequentialBaseFloatMatrixReader matrix_reader(rspecifier);
      for (; !matrix_reader.Done(); matrix_reader.Next())
        // Calls to Key and Value needs to happen at same time (Sequential reader)
        // Kaldi's alignments are len(feats) 
        ali_writer.Write(matrix_reader.Key(), std::vector<int32>(matrix_reader.Value().NumRows(), value));
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


