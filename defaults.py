class argHandler(dict):
    #A super duper fancy custom made CLI argument handler!!
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _descriptions = {'help, --h, -h': 'show this super helpful message and exit'}

    def setDefaults(self):

        """
        General options:

        --ffmpeg=<ffmpeg>         Specify which `ffmpeg` to use.
        --verbose                 Show processing progress.

        Face tracking options (track):

        <video>                   Path to video file.
        <shot.json>               Path to shot segmentation result file.
        <tracking>                Path to tracking result file.

        --min-size=<ratio>        Approximate size (in video height ratio) of the
                                    smallest face that should be detected. Default is
                                    to try and detect any object [default: 0.0].
        --every=<seconds>         Only apply detection every <seconds> seconds.
                                    Default is to process every frame [default: 0.0].
        --min-overlap=<ratio>     Associates face with tracker if overlap is greater
                                    than <ratio> [default: 0.5].
        --min-confidence=<float>  Reset trackers with confidence lower than <float>
                                    [default: 10.].
        --max-gap=<float>         Bridge gaps with duration shorter than <float>
                                    [default: 1.].

        Feature extraction options (features):

        <video>                   Path to video file.
        <tracking>                Path to tracking result file.
        <landmark_model>          Path to dlib facial landmark detection model.
        <embedding_model>         Path to dlib feature extraction model.
        <landmarks>               Path to facial landmarks detection result file.
        <embeddings>              Path to feature extraction result file.

        Visualization options (demo):

        <video>                   Path to video file.
        <tracking>                Path to tracking result file.
        <output>                  Path to demo video file.

        --height=<pixels>         Height of demo video file [default: 400].
        --from=<sec>              Encode demo from <sec> seconds [default: 0].
        --until=<sec>             Encode demo until <sec> seconds.
        --shift=<sec>             Shift result files by <sec> seconds [default: 0].
        --landmark=<path>         Path to facial landmarks detection result file.
        --label=<path>            Path to track identification result file.

        """
        self.define('track', False, '')
        self.define('--ffmpeg', None, '')
        self.define('--min-size', '0.0', '')
        self.define('--min-overlap', '0.5', '')
        self.define('--shift', '0', '')
        self.define('--landmark', None, '')
        self.define('--max-gap', '1.', '')
        self.define('--every', '0.2', '')
        self.define('--until', None, '')
        self.define('--from', '0', '')
        self.define('--min-confidence', '10.', '')
        self.define('--height', None, '')
        self.define('--verbose', True, '')
        
        #define the video, remaining paths will be saved in the same location
        self.define('<video>', 'sample/P2_S5_C2.avi', '')
        #self.define('<video>', 'sample/TheBigBangTheory.mkv', '')
        self.define('prefix', self['<video>'][:-4], '')
        self.define('<shot.json>', self['prefix'] + '.shots.json', '')
        self.define('<tracking>', self['prefix'] + '.track.txt', '')
        self.define('extract', False, '')
        self.define('<landmark_model>', 'dlib-models/shape_predictor_68_face_landmarks.dat', '')
        self.define('<embedding_model>', 'dlib-models/dlib_face_recognition_resnet_model_v1.dat', '')
        self.define('<landmarks>', self['prefix'] + '.landmarks.txt', '')
        self.define('<embeddings>', self['prefix'] + '.embedding.txt', '')
        self.define('--label', self['prefix'] + '.labels.txt', '')
        self.define('demo', True, '')
        self.define('<output>', self['prefix'] + '.track.mp4', '')
        self.define('--help', False, '')


    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description

    def help(self):
        print('Arguments:')
        spacing = max([len(i) for i in self._descriptions.keys()]) + 2
        for item in self._descriptions:
            currentSpacing = spacing - len(item)
            print('  --' + item + (' ' * currentSpacing) + self._descriptions[item])
        print('')
        exit()

    def parseArgs(self, args):
        print('')
        i = 1
        while i < len(args):
            if args[i] == '-h' or args[i] == '--h' or args[i] == '--help':
                self.help() #Time for some self help! :)
            if len(args[i]) < 2:
                print('ERROR - Invalid argument: ' + args[i])
                print('Try running flow --help')
                exit()
            argumentName = args[i][2:]
            if isinstance(self.get(argumentName), bool):
                if not (i + 1) >= len(args) and (args[i + 1].lower() != 'false' and args[i + 1].lower() != 'true') and not args[i + 1].startswith('--'):
                    print('ERROR - Expected boolean value (or no value) following argument: ' + args[i])
                    print('Try running flow --help')
                    exit()
                elif not (i + 1) >= len(args) and (args[i + 1].lower() == 'false' or args[i + 1].lower() == 'true'):
                    self[argumentName] = (args[i + 1].lower() == 'true')
                    i += 1
                else:
                    self[argumentName] = True
            elif args[i].startswith('--') and not (i + 1) >= len(args) and not args[i + 1].startswith('--') and argumentName in self:
                if isinstance(self[argumentName], float):
                    try:
                        args[i + 1] = float(args[i + 1])
                    except:
                        print('ERROR - Expected float for argument: ' + args[i])
                        print('Try running flow --help')
                        exit()
                elif isinstance(self[argumentName], int):
                    try:
                        args[i + 1] = int(args[i + 1])
                    except:
                        print('ERROR - Expected int for argument: ' + args[i])
                        print('Try running flow --help')
                        exit()
                self[argumentName] = args[i + 1]
                i += 1
            else:
                print('ERROR - Invalid argument: ' + args[i])
                print('Try running flow --help')
                exit()
            i += 1