// nnet3/nnet-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2015    Xiaohui Zhang

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

#include "nnet3/nnet-training.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetTrainer::NnetTrainer(const NnetTrainerOptions &config,
                         Nnet *nnet):
    config_(config),
    nnet_(nnet),
    compiler_(*nnet, config_.optimize_config, config_.compiler_config),
    num_minibatches_processed_(0),
    srand_seed_(RandInt(0, 100000)) {
  if (config.zero_component_stats)
    ZeroComponentStats(nnet);
  KALDI_ASSERT(config.momentum >= 0.0 &&
               config.max_param_change >= 0.0 &&
               config.backstitch_training_interval > 0);
  delta_nnet_ = nnet_->Copy();
  ScaleNnet(0.0, delta_nnet_);
  const int32 num_updatable = NumUpdatableComponents(*delta_nnet_);
  num_max_change_per_component_applied_.resize(num_updatable, 0);
  num_max_change_global_applied_ = 0;

  if (config_.read_cache != "") {
    bool binary;
    Input ki;
    if (ki.Open(config_.read_cache, &binary)) {
      compiler_.ReadCache(ki.Stream(), binary);
      KALDI_LOG << "Read computation cache from " << config_.read_cache;
    } else {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  }
}


void NnetTrainer::Train(const NnetExample &eg) {
  bool need_model_derivative = true;
  ComputationRequest request;
  GetComputationRequest(*nnet_, eg, need_model_derivative,
                        config_.store_component_stats,
                        &request);
  const NnetComputation *computation = compiler_.Compile(request);

  if (config_.backstitch_training_scale > 0.0 &&
      num_minibatches_processed_ % config_.backstitch_training_interval ==
      srand_seed_ % config_.backstitch_training_interval) {
    // backstitch training is incompatible with momentum > 0
    KALDI_ASSERT(config_.momentum == 0.0);
    FreezeNaturalGradient(true, delta_nnet_);
    bool is_backstitch_step1 = true;
    srand(srand_seed_ + num_minibatches_processed_);
    ResetGenerators(nnet_);
    TrainInternalBackstitch(eg, *computation, is_backstitch_step1);
    FreezeNaturalGradient(false, delta_nnet_); // un-freeze natural gradient
    is_backstitch_step1 = false;
    srand(srand_seed_ + num_minibatches_processed_);
    ResetGenerators(nnet_);
    TrainInternalBackstitch(eg, *computation, is_backstitch_step1);
  } else { // conventional training
    TrainInternal(eg, *computation);
  }

  num_minibatches_processed_++;
}

void NnetTrainer::TrainInternal(const NnetExample &eg,
                                const NnetComputation &computation) {
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.  This is mainly important for memory-norm.
  NnetComputer computer(config_.compute_config, computation,
                        nnet_, delta_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.io);
  computer.Run();

  this->ProcessOutputs(false, eg, &computer);
  computer.Run();

  // If relevant, add in the part of the gradient that comes from L2
  // regularization.
  ApplyL2Regularization(*nnet_,
                        GetNumNvalues(eg.io, false) * config_.l2_regularize_factor,
                        delta_nnet_);

  // Update the parameters of nnet
  bool success = UpdateNnetWithMaxChange(*delta_nnet_, config_.max_param_change,
      1.0, 1.0 - config_.momentum, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(config_.batchnorm_stats_scale, nnet_);

  // The following will only do something if we have a LinearComponent
  // or AffineComponent with orthonormal-constraint set to a nonzero value.
  ConstrainOrthonormal(nnet_);

  // Scale deta_nnet
  if (success)
    ScaleNnet(config_.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);
}

void NnetTrainer::TrainInternalBackstitch(const NnetExample &eg,
                                          const NnetComputation &computation,
                                          bool is_backstitch_step1) {
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.  This is mainly important for memory-norm.
  NnetComputer computer(config_.compute_config, computation,
                        nnet_, delta_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.io);
  computer.Run();

  bool is_backstitch_step2 = !is_backstitch_step1;
  this->ProcessOutputs(is_backstitch_step2, eg, &computer);
  computer.Run();

  BaseFloat max_change_scale, scale_adding;
  if (is_backstitch_step1) {
    // max-change is scaled by backstitch_training_scale;
    // delta_nnet is scaled by -backstitch_training_scale when added to nnet;
    max_change_scale = config_.backstitch_training_scale;
    scale_adding = -config_.backstitch_training_scale;
  } else {
    // max-change is scaled by 1 +  backstitch_training_scale;
    // delta_nnet is scaled by 1 + backstitch_training_scale when added to nnet;
    max_change_scale = 1.0 + config_.backstitch_training_scale;
    scale_adding = 1.0 + config_.backstitch_training_scale;
  }

  // If relevant, add in the part of the gradient that comes from L2
  // regularization.  It may not be optimally inefficient to do it on both
  // passes of the backstitch, like we do here, but it probably minimizes
  // any harmful interactions with the max-change.
  ApplyL2Regularization(*nnet_,
                        scale_adding * GetNumNvalues(eg.io, false) *
                        config_.l2_regularize_factor,
                        delta_nnet_);

  // Updates the parameters of nnet
  UpdateNnetWithMaxChange(*delta_nnet_, config_.max_param_change,
      max_change_scale, scale_adding, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  if (is_backstitch_step1) {
    // The following will only do something if we have a LinearComponent or
    // AffineComponent with orthonormal-constraint set to a nonzero value. We
    // choose to do this only on the 1st backstitch step, for efficiency.
    ConstrainOrthonormal(nnet_);
  }

  if (!is_backstitch_step1) {
    // Scale down the batchnorm stats (keeps them fresh... this affects what
    // happens when we use the model with batchnorm test-mode set).  Do this
    // after backstitch step 2 so that the stats are scaled down before we start
    // the next minibatch.
    ScaleBatchnormStats(config_.batchnorm_stats_scale, nnet_);
  }

  ScaleNnet(0.0, delta_nnet_);
}

void NnetTrainer::ProcessOutputs(bool is_backstitch_step2,
                                 const NnetExample &eg,
                                 NnetComputer *computer) {
  // normally the eg will have just one output named 'output', but
  // we don't assume this.
  // In backstitch training, the output-name with the "_backstitch" suffix is
  // the one computed after the first, backward step of backstitch.
  const std::string suffix = (is_backstitch_step2 ? "_backstitch" : "");

  if (config_.decouple) {
    // TODO: UpdateStats with percentage or number of removed frames
    // If training with decoupling then we will have a second output output_b
    std::string output_name = "output";
    std::string output_name_b = "output_b";

    int32 node_index = nnet_->GetNodeIndex(output_name);
    int32 node_index_b = nnet_->GetNodeIndex(output_name_b);
    KALDI_ASSERT(node_index >= 0);
    KALDI_ASSERT(node_index_b >= 0);

    // We precompute outputs from each and compute a mask
    // and then pass both to a special ComputeObjectiveFunction 
    // to avoid calling GetOutput twice
    const CuMatrixBase<BaseFloat> &output = computer->GetOutput(output_name);
    const CuMatrixBase<BaseFloat> &output_b = computer->GetOutput(output_name_b);

    // This uses a custom version of FindRowMaxId that takes CuVectors instead of CuArrays
    // We need CuVectors later for EqualElementMask
    // We extract SubVectors so that we don't have to copy Vectors into max_ids_mat later on
    CuMatrix<BaseFloat> max_ids_mat(1, output.NumRows(), kUndefined); 
    CuMatrix<BaseFloat> max_ids_mat_b(1, output.NumRows(), kUndefined);

    CuSubVector<BaseFloat> max_ids = max_ids_mat.Row(0);
    CuSubVector<BaseFloat> max_ids_b = max_ids_mat_b.Row(0);

    // Perform argmax for each example (by row)
    output.FindRowMaxId(&max_ids);
    output_b.FindRowMaxId(&max_ids_b);

    // Uses custom version of EqualElementMask that returns 1 for unequal items
    // Saves more computation
    CuMatrix<BaseFloat> unequal_mask(1, output.NumRows(), kUndefined);
    max_ids_mat.UnequalElementMask(max_ids_mat_b, &unequal_mask); 
    CuSubVector<BaseFloat> mask = unequal_mask.Row(0); // Doesn't copy
    int32 total_unequal = mask.Sum();
    int32 batchsize = mask.Dim();
    decouple_info_.UpdateStats(config_.print_interval,
                               num_minibatches_processed_,
                               total_unequal, batchsize);

    // Call special ComputeObjectiveFunction here for both outputs with mask
    std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
        end = eg.io.end();
    for (; iter != end; ++iter) {
      const NnetIo &io = *iter;
      int32 node_index = nnet_->GetNodeIndex(io.name);
      KALDI_ASSERT(node_index >= 0);
      if (nnet_->IsOutputNode(node_index)) {
        // either "output" or "output_b"
        KALDI_ASSERT(io.name == "output" || io.name == "output_b");

        ObjectiveType obj_type = nnet_->GetNode(node_index).u.objective_type;
        BaseFloat tot_weight, tot_objf;
        bool supply_deriv = true;
        if (io.name == "output")
          ComputeObjectiveFunctionMasked(io.features, obj_type, io.name,
                                         output, mask,
                                         supply_deriv, computer,
                                         &tot_weight, &tot_objf);
        else if (io.name == "output_b")
          ComputeObjectiveFunctionMasked(io.features, obj_type, io.name,
                                         output_b, mask,
                                         supply_deriv, computer,
                                         &tot_weight, &tot_objf);

        objf_info_[io.name + suffix].UpdateStats(io.name + suffix,
                                        config_.print_interval,
                                        num_minibatches_processed_,
                                        tot_weight, tot_objf);
      }
    }

    /* CuMatrix<BaseFloat> test_output_deriv(output.NumRows(), 5, kUndefined); */
    /* test_output_deriv.SetRandn(); */
    /* CuMatrix<BaseFloat> new_output_deriv(output.NumRows(), 5, kSetZero); */
    /* new_output_deriv.AddDiagVecMat(1.0, mask, test_output_deriv, kNoTrans, 0.0); */

    
    // UpdateStats

    // PSEUDOCODE END
  } else { // Normal operation
    std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
        end = eg.io.end();
    for (; iter != end; ++iter) {
      const NnetIo &io = *iter;
      // loops across all potential input and output nodes (hence check for outputNode below)
      //
      int32 node_index = nnet_->GetNodeIndex(io.name);
      KALDI_ASSERT(node_index >= 0);

      if (nnet_->IsOutputNode(node_index)) {

        /* KALDI_LOG << "Computing for io.name " << io.name; */
        ObjectiveType obj_type = nnet_->GetNode(node_index).u.objective_type;
        BaseFloat tot_weight, tot_objf;
        bool supply_deriv = true;
        ComputeObjectiveFunction(io.features, obj_type, io.name,
                                 supply_deriv, computer,
                                 &tot_weight, &tot_objf);
        // KALDI_LOG << "tot_weight: " << tot_weight;
        objf_info_[io.name + suffix].UpdateStats(io.name + suffix,
                                        config_.print_interval,
                                        num_minibatches_processed_,
                                        tot_weight, tot_objf);
      }
    }
  }
}

bool NnetTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  std::vector<std::pair<std::string, const ObjectiveFunctionInfo*> > all_pairs;
  for (; iter != end; ++iter)
    all_pairs.push_back(std::pair<std::string, const ObjectiveFunctionInfo*>(
        iter->first, &(iter->second)));
  // ensure deterministic order of these names (this will matter in situations
  // where a script greps for the objective from the log).
  std::sort(all_pairs.begin(), all_pairs.end());
  bool ans = false;
  for (size_t i = 0; i < all_pairs.size(); i++) {
    const std::string &name = all_pairs[i].first;
    const ObjectiveFunctionInfo &info = *(all_pairs[i].second);
    bool ok = info.PrintTotalStats(name);
    ans = ans || ok;
  }
  PrintMaxChangeStats();
  if (config_.decouple) {
    decouple_info_.PrintTotalStats();
  }
  return ans;
}

void NnetTrainer::PrintMaxChangeStats() const {
  KALDI_ASSERT(delta_nnet_ != NULL);
  int32 i = 0;
  for (int32 c = 0; c < delta_nnet_->NumComponents(); c++) {
    Component *comp = delta_nnet_->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
                  << "UpdatableComponent; change this code.";
      if (num_max_change_per_component_applied_[i] > 0)
        KALDI_LOG << "For " << delta_nnet_->GetComponentName(c)
                  << ", per-component max-change was enforced "
                  << (100.0 * num_max_change_per_component_applied_[i]) /
                     (num_minibatches_processed_ *
                     (config_.backstitch_training_scale == 0.0 ? 1.0 :
                     1.0 + 1.0 / config_.backstitch_training_interval))
                  << " \% of the time.";
      i++;
    }
  }
  if (num_max_change_global_applied_ > 0)
    KALDI_LOG << "The global max-change was enforced "
              << (100.0 * num_max_change_global_applied_) /
                 (num_minibatches_processed_ *
                 (config_.backstitch_training_scale == 0.0 ? 1.0 :
                 1.0 + 1.0 / config_.backstitch_training_interval))
              << " \% of the time.";
}

void ObjectiveFunctionInfo::UpdateStats(
    const std::string &output_name,
    int32 minibatches_per_phase,
    int32 minibatch_counter,
    BaseFloat this_minibatch_weight,
    BaseFloat this_minibatch_tot_objf,
    BaseFloat this_minibatch_tot_aux_objf) {
  int32 phase = minibatch_counter / minibatches_per_phase;
  if (phase != current_phase) {
    KALDI_ASSERT(phase > current_phase);
    PrintStatsForThisPhase(output_name, minibatches_per_phase,
                           phase);
    current_phase = phase;
    tot_weight_this_phase = 0.0;
    tot_objf_this_phase = 0.0;
    tot_aux_objf_this_phase = 0.0;
    minibatches_this_phase = 0;
  }
  minibatches_this_phase++;
  /* KALDI_LOG << "tot_weight_this_phase: " << tot_weight_this_phase << " this_minibatch_weight: " << this_minibatch_weight; */
  tot_weight_this_phase += this_minibatch_weight;
  tot_objf_this_phase += this_minibatch_tot_objf;
  tot_aux_objf_this_phase += this_minibatch_tot_aux_objf;
  tot_weight += this_minibatch_weight;
  tot_objf += this_minibatch_tot_objf;
  tot_aux_objf += this_minibatch_tot_aux_objf;
}

void DecoupleInfo::UpdateStats(
    int32 minibatches_per_phase,
    int32 minibatch_counter,
    int32 this_num_unequal,
    int32 this_minibatch_size) {
  latest_num_unequal = this_num_unequal;
  latest_minibatch_size = this_minibatch_size;

  int32 phase = minibatch_counter / minibatches_per_phase;
  if (phase != current_phase) {
    KALDI_ASSERT(phase > current_phase);
    PrintStatsForThisPhase(minibatches_per_phase,
                           phase);
    current_phase = phase;
    minibatches_this_phase = 0;
    tot_num_unequal_this_phase = 0;
  }
  minibatches_this_phase++;
  tot_num_unequal_this_phase += this_num_unequal;
  tot_num_unequal += this_num_unequal;
  minibatches++;
}

void DecoupleInfo::PrintStatsForThisPhase(
    int32 minibatches_per_phase,
    int32 phase) const {
  int32 start_minibatch = current_phase * minibatches_per_phase,
      end_minibatch = phase * minibatches_per_phase - 1;

  if (minibatches_per_phase == minibatches_this_phase) {
    BaseFloat average_num_unequal = tot_num_unequal_this_phase / (minibatches_this_phase);
    KALDI_LOG << "Average number of unequal / disagreeing frames " 
              << "for minibatches " << start_minibatch
              << '-' << end_minibatch << " (" << minibatches_this_phase << ") is "
              << average_num_unequal << " (latest number is " << latest_num_unequal 
              << " from a minibatch of " << latest_minibatch_size << " frames.)";
  }
}

void DecoupleInfo::PrintTotalStats() const {
  BaseFloat avg_unequal = tot_num_unequal / minibatches;
  KALDI_LOG << "Overall average number of unequal / disagreeing frames is "
            << avg_unequal << " over " << minibatches << " minibatches.";
  KALDI_LOG << "[this line is to be parsed by a script:] "
            << "num-unequal-per-minibatch="
            << avg_unequal;
}


void ObjectiveFunctionInfo::PrintStatsForThisPhase(
    const std::string &output_name,
    int32 minibatches_per_phase,
    int32 phase) const {
  int32 start_minibatch = current_phase * minibatches_per_phase,
      end_minibatch = phase * minibatches_per_phase - 1;

  if (tot_aux_objf_this_phase == 0.0) {
    if (minibatches_per_phase == minibatches_this_phase) {
      KALDI_LOG << "Average objective function for '" << output_name
                << "' for minibatches " << start_minibatch
                << '-' << end_minibatch << " is "
                << (tot_objf_this_phase / tot_weight_this_phase) << " over "
                << tot_weight_this_phase << " frames.";
    } else {
      KALDI_LOG << "Average objective function for '" << output_name
                << " using " << minibatches_this_phase
                << " minibatches in minibatch range " << start_minibatch
                << '-' << end_minibatch << " is "
                << (tot_objf_this_phase / tot_weight_this_phase) << " over "
                << tot_weight_this_phase << " frames.";
    }
  } else {
    BaseFloat objf = (tot_objf_this_phase / tot_weight_this_phase),
        aux_objf = (tot_aux_objf_this_phase / tot_weight_this_phase),
        sum_objf = objf + aux_objf;
    if (minibatches_per_phase == minibatches_this_phase) {
      KALDI_LOG << "Average objective function for '" << output_name
                << "' for minibatches " << start_minibatch
                << '-' << end_minibatch << " is "
                << objf << " + " << aux_objf << " = " << sum_objf
                << " over " << tot_weight_this_phase << " frames.";
    } else {
      KALDI_LOG << "Average objective function for '" << output_name
                << "' using " << minibatches_this_phase
                << " minibatches in  minibatch range " << start_minibatch
                << '-' << end_minibatch << " is "
                << objf << " + " << aux_objf << " = " << sum_objf
                << " over " << tot_weight_this_phase << " frames.";
    }
  }
}

bool ObjectiveFunctionInfo::PrintTotalStats(const std::string &name) const {
  BaseFloat objf = (tot_objf / tot_weight),
        aux_objf = (tot_aux_objf / tot_weight),
        sum_objf = objf + aux_objf;
  if (tot_aux_objf == 0.0) {
    KALDI_LOG << "Overall average objective function for '" << name << "' is "
              << (tot_objf / tot_weight) << " over " << tot_weight << " frames.";
  } else {
    KALDI_LOG << "Overall average objective function for '" << name << "' is "
              << objf << " + " << aux_objf << " = " << sum_objf
              << " over " << tot_weight << " frames.";
  }
  KALDI_LOG << "[this line is to be parsed by a script:] "
            << "log-prob-per-frame="
            << objf;
  return (tot_weight != 0.0);
}

NnetTrainer::~NnetTrainer() {
  if (config_.write_cache != "") {
    Output ko(config_.write_cache, config_.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), config_.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << config_.write_cache;
  }
  delete delta_nnet_;
}

void ComputeObjectiveFunction(const GeneralMatrix &supervision,
                              ObjectiveType objective_type,
                              const std::string &output_name,
                              bool supply_deriv,
                              NnetComputer *computer,
                              BaseFloat *tot_weight,
                              BaseFloat *tot_objf) {
  const CuMatrixBase<BaseFloat> &output = computer->GetOutput(output_name);

  if (output.NumCols() != supervision.NumCols())
    KALDI_ERR << "Nnet versus example output dimension (num-classes) "
              << "mismatch for '" << output_name << "': " << output.NumCols()
              << " (nnet) vs. " << supervision.NumCols() << " (egs)\n";

  switch (objective_type) {
    case kLinear: {
      // objective is x * y.
      switch (supervision.Type()) {
        case kSparseMatrix: {
          const SparseMatrix<BaseFloat> &post = supervision.GetSparseMatrix();
          CuSparseMatrix<BaseFloat> cu_post(post);
          // The cross-entropy objective is computed by a simple dot product,
          // because after the LogSoftmaxLayer, the output is already in the form
          // of log-likelihoods that are normalized to sum to one.
          *tot_weight = cu_post.Sum();
          // KALDI_LOG << "tot_weight in Compute: " << *tot_weight;
          // KALDI_LOG << "tot_weight in Compute numrows: " << cu_post.NumRows() << " numcols: " << cu_post.NumCols();
          *tot_objf = TraceMatSmat(output, cu_post, kTrans);
          if (supply_deriv) {
            CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols(),
                                             kUndefined);
            cu_post.CopyToMat(&output_deriv);
            computer->AcceptInput(output_name, &output_deriv);
          }
          break;
        }
        case kFullMatrix: {
          // there is a redundant matrix copy in here if we're not using a GPU
          // but we don't anticipate this code branch being used in many cases.
          CuMatrix<BaseFloat> cu_post(supervision.GetFullMatrix());
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptInput(output_name, &cu_post);
          break;
        }
        case kCompressedMatrix: {
          Matrix<BaseFloat> post;
          supervision.GetMatrix(&post);
          CuMatrix<BaseFloat> cu_post;
          cu_post.Swap(&post);
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptInput(output_name, &cu_post);
          break;
        }
      }
      break;
    }
    case kQuadratic: {
      // objective is -0.5 (x - y)^2
      CuMatrix<BaseFloat> diff(supervision.NumRows(),
                               supervision.NumCols(),
                               kUndefined);
      diff.CopyFromGeneralMat(supervision);
      diff.AddMat(-1.0, output);
      *tot_weight = diff.NumRows();
      *tot_objf = -0.5 * TraceMatMat(diff, diff, kTrans);
      if (supply_deriv)
        computer->AcceptInput(output_name, &diff);
      break;
    }
    default:
      KALDI_ERR << "Objective function type " << objective_type
                << " not handled.";
  }
}

void ComputeObjectiveFunctionMasked(const GeneralMatrix &supervision,
                                    ObjectiveType objective_type,
                                    const std::string &output_name,
                                    const CuMatrixBase<BaseFloat> &output,
                                    const CuSubVector<BaseFloat> &mask,
                                    bool supply_deriv,
                                    NnetComputer *computer,
                                    BaseFloat *tot_weight,
                                    BaseFloat *tot_objf) {
  /* const CuMatrixBase<BaseFloat> &output = computer->GetOutput(output_name); */

  if (output.NumCols() != supervision.NumCols())
    KALDI_ERR << "Nnet versus example output dimension (num-classes) "
              << "mismatch for '" << output_name << "': " << output.NumCols()
              << " (nnet) vs. " << supervision.NumCols() << " (egs)\n";

  switch (objective_type) {
    case kLinear: {
      // objective is x * y.
      switch (supervision.Type()) {
        case kSparseMatrix: {
          const SparseMatrix<BaseFloat> &post = supervision.GetSparseMatrix();
          CuSparseMatrix<BaseFloat> cu_post(post);
          // The cross-entropy objective is computed by a simple dot product,
          // because after the LogSoftmaxLayer, the output is already in the form
          // of log-likelihoods that are normalized to sum to one.
          *tot_weight = cu_post.Sum();
          // KALDI_LOG << "tot_weight in Compute: " << *tot_weight;
          // KALDI_LOG << "tot_weight in Compute numrows: " << cu_post.NumRows() << " numcols: " << cu_post.NumCols();
          *tot_objf = TraceMatSmat(output, cu_post, kTrans);
          if (supply_deriv) {
            CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols(),
                                             kUndefined);
            cu_post.CopyToMat(&output_deriv);
            computer->AcceptInput(output_name, &output_deriv);
          }
          break;
        }
        case kFullMatrix: {
          // there is a redundant matrix copy in here if we're not using a GPU
          // but we don't anticipate this code branch being used in many cases.
          CuMatrix<BaseFloat> cu_post(supervision.GetFullMatrix());
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptInput(output_name, &cu_post);
          break;
        }
        case kCompressedMatrix: {
          Matrix<BaseFloat> post;
          supervision.GetMatrix(&post);
          CuMatrix<BaseFloat> cu_post;
          cu_post.Swap(&post);
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptInput(output_name, &cu_post);
          break;
        }
      }
      break;
    }
    case kQuadratic: {
      // objective is -0.5 (x - y)^2
      CuMatrix<BaseFloat> diff(supervision.NumRows(),
                               supervision.NumCols(),
                               kUndefined);
      diff.CopyFromGeneralMat(supervision);
      diff.AddMat(-1.0, output);
      *tot_weight = diff.NumRows();
      *tot_objf = -0.5 * TraceMatMat(diff, diff, kTrans);
      if (supply_deriv)
        computer->AcceptInput(output_name, &diff);
      break;
    }
    default:
      KALDI_ERR << "Objective function type " << objective_type
                << " not handled.";
  }
}

} // namespace nnet3
} // namespace kaldi
