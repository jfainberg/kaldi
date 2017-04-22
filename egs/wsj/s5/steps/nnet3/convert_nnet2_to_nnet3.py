#!/usr/bin/env python
# Copyright 2016

# This script converts nnet2 into nnet3 models.
# It requires knowledge of valid components.
# These can be modified in the configuration section below.


## TODO:
# Change reading to read entire file
# Change way we parse to using str.find('[', start, end)
# Use enumerate(list): matrix.append(linparams + ' ' + bias[i]) to create matrices

# and read using a generator:
# http://stackoverflow.com/questions/7421621/python-read-through-file-until-match-read-until-next-pattern

import argparse, subprocess, os, tempfile, logging, sys, shutil, fileinput, re
from collections import namedtuple, defaultdict

# Begin configuration section
# Components and their corresponding node names

NODE_NAMES = {
  "<AffineComponent>":"affine",
  "<AffineComponentPreconditioned>":"affine",
  "<AffineComponentPreconditionedOnline>":"affine",
  "<BlockAffineComponent>":"affine",
  "<BlockAffineComponentPreconditioned>":"affine",
  "<SigmoidComponent>":"nonlin",
  "<TanhComponent>":"nonlin",
  "<PowerComponent>":"nonlin",
  "<RectifiedLinearComponent>":"nonlin",
  "<SoftHingeComponent>":"nonlin",
  "<PnormComponent>":"nonlin",
  "<NormalizeComponent>":"renorm",
  "<MaxoutComponent>":"maxout",
  "<MaxpoolingComponent>":"maxpool",
  "<ScaleComponent>":"rescale",
  "<DropoutComponent>":"dropout",
  "<SoftmaxComponent>":"softmax",
  "<LogSoftmaxComponent>":"log-softmax",
  "<FixedScaleComponent>":"fixed-scale",
  "<FixedAffineComponent>":"fixed-affine",
  "<FixedLinearComponent>":"fixed-linear",
  "<FixedBiasComponent>":"fixed-bias",
  "<PermuteComponent>":"permute",
  "<AdditiveNoiseComponent>":"noise",
  "<Convolutional1dComponent>":"conv",
  "<SumGroupComponent>":"sum-group",
  "<DctComponent>":"dct",
  "<SpliceComponent>":"splice",
  "<SpliceMaxComponent>":"splice"
}

KNOWN_COMPONENTS = NODE_NAMES.keys()
# End configuration section

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def GetArgs():
  parser = argparse.ArgumentParser(description="Converts nnet2 into nnet3 models.",
                                   epilog="e.g. steps/convert_nnet2_to_nnet3.py exp/tri4_nnet2 exp/tri4_nnet3");
  parser.add_argument("--tmpdir", type=str,
                      help="Set a custom location for the temporary directory.")
  parser.add_argument("--skip-cleanup", action='store_true',
                      help="Will not remove the temporary directory.")
  parser.add_argument("--model", type=str, default='final.mdl',
                      help="Choose a specific model to convert (default: final.mdl).")
  parser.add_argument("nnet2_dir", metavar="src-nnet2-dir", type=str,
                      help="")
  parser.add_argument("nnet3_dir", metavar="src-nnet3-dir", type=str,
                      help="")

  print(' '.join(sys.argv))

  args = parser.parse_args()

  # Check arguments.
  if not os.path.exists(args.nnet3_dir):
    os.makedirs(args.nnet3_dir)
  if args.tmpdir and not os.path.exists(args.tmpdir):
    os.makedirs(args.tmpdir)

  return args

def IsComponent(component):
  # Recognises opening tags "<Component>"
  # *not* closing tags "</Component>"
  if component in KNOWN_COMPONENTS:
    return True
  else:
    if "Component" in component and '/' not in component:
      logger.warning('Assuming "{0}" is not a component'.format(component))
    return False

def ExpectToken(expect, token):
  if expect != token:
    logger.error('Unexpected token, expected "{0}", got "{1}". Quitting.'.format(expect, token))
    sys.exit()

def ConsumeToken(token, line):
  '''Returns line without token'''
  ExpectToken(token, line.split()[0])
  return line.partition(token)[2]

def MakeSpliceString(nodename, context):
  '''
  E.g. MakeSpliceString('renorm4', [-4, 4])
  returns 'Append(Offset(renorm4, -4), Offset(renorm4, 4))'
  '''
  assert type(context) == list, 'MakeSpliceString: context argument must be a list'
  string = ['Offset({0}, {1})'.format(nodename, i) for i in context]
  string = 'Append(' + ', '.join(string) + ')'
  return string

def Read(args):
  # args.tmpdir = None if not set
  tmpdir = tempfile.mkdtemp(dir=args.tmpdir) 

  # Convert nnet2 model to text
  # We replace AffineComponentPreconditioned with AffineComponent,
  # as the former is not in nnet3
  result = subprocess.call('nnet-am-copy --remove-preconditioning=true --binary=false {0}/{1} {2}/{1}'.format(args.nnet2_dir, args.model, args.tmpdir), shell=True)
  if (result != 0):
    raise OSError('Could not run nnet-am-copy. Did you source path.sh?')

  # Read nnet2 acoustic model and write components to tmpdir/*
  with open(os.path.join(args.tmpdir, args.model)) as f:
    # Transition model
    line = f.readline()
    ExpectToken('<TransitionModel>', line.strip())
    with open(os.path.join(tmpdir, 'transition_model'), 'w') as fc:
      fc.write(line)
      for line in f:
        if line.startswith('<Nnet>'):
          break
        fc.write(line)
      
    # Nnet header
    line = ConsumeToken('<Nnet>', line)
    line = ConsumeToken('<NumComponents>', line)
    num_components = line.split()[0]
    line = line.partition(num_components)[2]
    line = ConsumeToken('<Components>', line)
    with open(os.path.join(tmpdir, 'num_components'), 'w') as fc:
      fc.write(num_components)
    
    tup = namedtuple('Component', 'name nodename')
    structure = [] # ordered list of named tuples
    counts = defaultdict(int) # node names counts, e.g. affine:2

    # First component
    current_component = line.split()[0]
    if IsComponent(current_component):
      node_name = NODE_NAMES[current_component]
      counts[node_name] += 1
      filename = node_name + str(counts[node_name])
      structure.append(tup(current_component, filename))
      fc = open(os.path.join(tmpdir, filename), 'w')
      fc.write(line)
      
    # Remaining components
    for line in f:
      # Swap files when encountering components or priors
      current_component = line.split()[0]
      if IsComponent(current_component):
        node_name = NODE_NAMES[current_component]
        counts[node_name] += 1
        filename = node_name + str(counts[node_name])
        structure.append(tup(current_component, filename))
        # New component, new file
        fc.close(); fc = open(os.path.join(tmpdir, filename), 'w')
      elif current_component == '</Components>':
        # Priors
        line = ConsumeToken('</Components>', line)
        line = ConsumeToken('</Nnet>', line)
        fc.close(); fc = open(os.path.join(tmpdir, 'priors'), 'w')
      # Continue writing
      #if '<P>' in line:
        # P-parameter not present in Pnorm components in nnet3
        #line = re.sub('<P> [0-9] ', '', line)
      fc.write(line)

    fc.close()

  return tmpdir, structure

def Write(args, tmpdir, structure):
  with open(os.path.join(tmpdir, 'nnet3.raw'), 'w') as f:
    ## Transition model 
    #for line in fileinput.input(os.path.join(tmpdir, 'transition_model')):
      #f.write(line)

    # Nnet header and configuration
    f.write('<Nnet3> \n')
    
    splice_components = ['<SpliceComponent>', '<SpliceMaxCompoment>']

    for i, component in enumerate(structure):
      # We construct the input string for the next node at the current node
      if component.name in splice_components:
        with open(os.path.join(tmpdir, component.nodename)) as fc:
          line = fc.readline()
          line = ConsumeToken(component.name, line)
          line = ConsumeToken('<InputDim>', line)
          [input_dim, _, line] = line.strip().partition(' ')
          line = ConsumeToken('<Context>', line)
          context = line.strip()[1:-1].split()
        if i == 0:
          f.write('input-node name=input dim={0}\n'.format(input_dim))
          input_string = MakeSpliceString('input', context) 
        else:
          input_string = MakeSpliceString(structure[i-1].nodename, context) 
      else:
        # Normal component
        f.write('component-node name={0} component={0} input={1}\n'.format(component.nodename, input_string))
        input_string = structure[i].nodename

    # Output: assume objective=linear
    f.write('output-node name=output input={0} objective=linear\n\n'.format(structure[-1].nodename))

    with open(os.path.join(tmpdir, 'num_components')) as fc:
      f.write('<NumComponents> {0}\n'.format(fc.read()))
      
    # Components
    for component in structure:
      if component.name not in splice_components:
        f.write('<ComponentName> {0} '.format(component.nodename))
        for line in fileinput.input(os.path.join(tmpdir, component.nodename)):
          f.write(line)
    f.write('</Nnet3>')

  # Compose TM and Nnet and get Left/Right Context 
  # This also works as a sanity check of the nnet
  subprocess.call('nnet3-am-init --binary=false {0} {1} {2}'.format(os.path.join(tmpdir, 'transition_model'),
                                                                    os.path.join(tmpdir, 'nnet3.raw'),
                                                                    os.path.join(tmpdir, 'nnet3_nopriors.mdl')), shell=True)

  # Priors
  #with open(os.path.join(tmpdir, 'priors')) as fc:
    #f.write('<Priors>{0}'.format(fc.read()))

#os.path.join(args.nnet3_dir, args.model)

    # LeftContext RightContext
    #left_context = 0; right_context = 0; #TODO
    #f.write('<LeftContext> {0} <RightContext> {1} '.format(left_context, right_context))
  

def Main():
  args = GetArgs()
  logger.info('Converting nnet2 model {0}/{1} to nnet3 model in {2}'.format(args.nnet2_dir, args.model, args.nnet3_dir))

  # Parse nnet2 model to individual text files in tmpdir.
  # This writes each component to files named by the
  # corresponding node name and number, e.g. "affine1",
  # as given in NODE_NAMES.
  # Each instance is added to the list 'structure'.
  [tmpdir, structure] = Read(args)
  print structure

  # Combine the text files in tmpdir to an nnet3 model.
  # SpliceComponents are converted to Descriptors for the 
  # succeeding component.
  Write(args, tmpdir, structure) 

    # traversing through them... ReadSpliceComponent?
    # SpliceComponent isn't necessarily first... (or at all?)
    #for var in variables:
      #variables[var] = re.findall('<' + var + '> ([-\d \[\]]*)', line)
      #if not variables[var]:
        #logger.error('Failed to extract information for {0}'.format(var))
        #sys.exit()
    
  if not args.skip_cleanup:
    shutil.rmtree(tmpdir)
  else:
    logger.info('Not removing temporary directory {0}'.format(tmpdir))
   
  logger.info('Wrote nnet3 model to {0}'.format(os.path.join(args.nnet3_dir, args.model)))

if __name__ == "__main__":
  Main()
