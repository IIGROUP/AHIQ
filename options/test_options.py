from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--test_ref_path', type=str, default='Test_Ref/', help='path to reference images')
        self._parser.add_argument('--test_dis_path', type=str, default='Test_Dis/', help='path to distortion images')
        self._parser.add_argument('--test_list', type=str, default='test.txt', help='training data')            
        
        self._parser.add_argument('--batch_size', type=int, default=10, help='input batch size')
        self._parser.add_argument('--test_file_name', type=str, default='results.txt', help='txt path to save results')
        self._parser.add_argument('--n_ensemble', type=int, default=20, help='crop method for test: five points crop or nine points crop or random crop for several times')
        self._parser.add_argument('--flip', type=bool, default=False, help='if flip images when testing')
        self._parser.add_argument('--resize', type=bool, default=False, help='if resize images when testing')
        self._parser.add_argument('--size', type=int, default=224, help='the resize shape')
        self.is_train = False
