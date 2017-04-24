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

class Nnet3Model(object):
  '''
  Holds configuration for an Nnet3 model 
  in a config style format.
  '''
  
  def __init__(self):
    self.input_dim = -1
    self.output_dim = -1
    self.counts = defaultdict(int)
    self.num_components = 0
    self.context = ''
    self.components = []

  def AddComponent(self, component, pairs):
    '''
    Adds components to model s.

    Most of the formatting happens here.
    '''
    self.num_components += 1

    # remove nnet2 specific tokens
    if component == '<PnormComponent>' and '<P>' in pairs:
      pairs.pop('<P>')

    # format pairs: {'<InputDim>':43} -> {'input-dim':43}
    pairs = ['{0}={1}'.format(TokenToString(key),pairs[key]) for key in pairs]
    
    # keep track of layer type number (e.g. affine3)
    node_name = NODE_NAMES[component]
    self.counts[node_name] += 1

    # e.g. affine3
    ident = node_name + str(self.counts[node_name])

    # <PnormComponent> -> PnormComponent
    component = component[1:-1]

    self.components.append((ident, component, pairs))

  def WriteConfig(self, filename):
    '''
    Write config to file filename.
    '''
    with open(filename, 'w') as f:

      for component in self.components:
        config_string = ' '.join(component[2])
        f.write('component name={name} type={comp_type} {config_string}\n'.format(name=component[0], comp_type=component[1], config_string=config_string))
          # component name=L0_fixaffine type=FixedAffineComponent matrix=exp/nnet3/tdnn/configs/lda.mat
          # component name=Tdnn_0_affine type=NaturalGradientAffineComponent input-dim=215 output-dim=1024  bias-stddev=0  max-change=0.75
          # component name=Tdnn_0_relu type=RectifiedLinearComponent dim=1024
          # component name=Tdnn_0_renorm type=NormalizeComponent dim=1024 target-rms=1.0

      f.write('\n# Component nodes\n')
      f.write('input-node name=input dim={0}\n'.format(self.input_dim))
      previous_component=MakeSpliceString('input', self.context)
      for component in self.components:
        f.write('component-node name={name} component={name} input={inp}\n'.format(name=component[0], inp=previous_component))
        previous_component = component[0]
      # component-node name=Tdnn_1_affine component=Tdnn_1_affine input=Append(Offset(Tdnn_0_renorm, -1) , Offset(Tdnn_0_renorm, 2))
      # component-node name=Tdnn_1_relu component=Tdnn_1_relu input=Tdnn_1_affine
      # component-node name=Tdnn_1_renorm component=Tdnn_1_renorm input=Tdnn_1_relu
      logger.warning('Assuming linear objective.')
      f.write('output-node name=output input={inp} objective={obj}\n'.format(inp=previous_component, obj='linear'))

def TokenToString(token):
  '''
  <InputDim> -> input-dim
  '''
  # remove <>
  string = token[1:-1]
  # InputDim to input-dim
  string = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r'-\1', string).lower()
  return string

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

  # Set up Nnet3 model object
  nnet3 = Nnet3Model()

  # Convert nnet2 model to text
  # We replace AffineComponentPreconditioned with AffineComponent,
  # as the former is not in nnet3
  result = subprocess.call('nnet-am-copy --remove-preconditioning=true --binary=false {0}/{1} {2}/{1}'.format(args.nnet2_dir, args.model, args.tmpdir), shell=True)
  if (result != 0):
    raise OSError('Could not run nnet-am-copy. Did you source path.sh?')

  # Read nnet2 acoustic model, write components to tmpdir/*, 
  # and a config to tmpdir/init_nnet3.config
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

    # Every time we see a component, we get the component name and
    # type, then we store each key-value pair (e.g. "<InputDim> 43")
    splice_components = ['<SpliceComponent>', '<SpliceMaxCompoment>']

    # First component
    current_component = line.split()[0]
    if IsComponent(current_component):
      node_name = NODE_NAMES[current_component]
      counts[node_name] += 1
      line = ConsumeToken(current_component, line)

      # if splice component then deal with it slightly differently
      if current_component in splice_components:
        line = ConsumeToken('<InputDim>', line)
        [input_dim, _, line] = line.strip().partition(' ')
        line = ConsumeToken('<Context>', line)
        context = line.strip()[1:-1].split()
        nnet3.input_dim = input_dim
        nnet3.context = context
        filename = node_name + str(counts[node_name])
      else:
      # get key-value pairs
      # '<InputDim> 2000 <OutputDim> 250 </PnormComponent>'  -->  [('<InputDim>', '2000'), ('<OutputDim>', '250')]
        pairs = re.findall('(<\w+>) (\w+)', line)
        nnet3.AddComponent(current_component, dict(pairs)
        filename = node_name + str(counts[node_name])
      #nnet3.AddComponent(current_component, pairs, filename)

      structure.append(tup(current_component, filename))
      # fc = open(os.path.join(tmpdir, filename), 'w')
      # fc.write(line)
      
    # Remaining components
    for line in f:
      # Swap files when encountering components or priors
      current_component = line.split()[0]
      if IsComponent(current_component):
        node_name = NODE_NAMES[current_component]
        counts[node_name] += 1
        filename = node_name + str(counts[node_name])
        structure.append(tup(current_component, filename))

        line = ConsumeToken(current_component, line)
        pairs = re.findall('(<\w+>) (\w+)', line)
        pairs = dict(pairs)
        nnet3.AddComponent(current_component, pairs)

        # New component, new file
        # fc.close(); fc = open(os.path.join(tmpdir, filename), 'w')
      elif current_component == '</Components>':
        # Priors
        line = ConsumeToken('</Components>', line)
        line = ConsumeToken('</Nnet>', line)
        fc.close(); fc = open(os.path.join(tmpdir, 'priors'), 'w')
      # Continue writing
      #if '<P>' in line:
        # P-parameter not present in Pnorm components in nnet3
        #line = re.sub('<P> [0-9] ', '', line)
      # fc.write(line)

    # fc.close()

    # check if we've read all components
    if num_components != nnet3.num_components:
      logger.error('Did not read all components succesfully: {0}/{1}'.format(nnet3.num_components, num_components))

    nnet3.WriteConfig(os.path.join(tmpdir, 'config'))

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
  # Write(args, tmpdir, structure) 

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
