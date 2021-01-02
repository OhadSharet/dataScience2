import numpy as np
from loadMNIST import MnistDataloader, show_images
from collections import Counter


class Ex3():
    BIT_SIZE = 252
    K = 10
    MAX_INT= 9223372036854775807
    BIT_AVG_THRESHOLD = 1/5
    MAX_REPS = 4


    def __init__(self):
        '''
        @:param clusters(list) : list of clusters for each number
        @:param group_rep (list) : list of group representative
        '''
        (self.img_train, self.img_test) = (None, None)
        self.clusters =[[] for _ in range(self.K)]
        self.load_data()
        self.clusters_rep = self._init_cluster_centers_manually()
        self.cluster_labels = [-1] * 10

    def _init_cluster_centers_randomly(self):
        ls = []
        for _ in range(self.K):
            ls.append(self._create_ranom_img())
        return ls

    def _init_cluster_centers_manually(self):
        ls = [1,6,189,149,336,219,13,15,265,362]
        ls_slim = [156,124,190,245,338,145,274,307,114,153]
        return [self.img_train[index].img for index in ls_slim]

    def _create_ranom_img(self):
        mat = [np.random.randint(2, size=28) for _ in range(28)]
        return np.array(mat)

    def run(self):
        self._train_model()
        self._test_model()

    def _test_model(self):
        print("testing model")
        self.clusters = [[] for _ in range(self.K)]
        self._devide_imgs_to_clusters(self.img_test)
        self._print_result()

    def _print_result(self):
        print("printing result")
        success_count_lst = self._measure_success()
        labels_to_print = []
        for i in range(self.K):
            sucsess_rate = -1
            if len(self.clusters[i])!=0:
                sucsess_rate = success_count_lst[i] / len(self.clusters[i])
            labels_to_print.append("clusrt '%s'|success '%s' " % (self.cluster_labels[i], format(sucsess_rate, ".3f")))
        for st in labels_to_print:
            print(st)
        show_images(self.clusters_rep,labels_to_print)

    def _measure_success(self):
        sucsess_count = [0] * self.K
        for i in range(self.K):
            for img in self.clusters[i]:
                if img.label == self.cluster_labels[i]:
                    sucsess_count[i] += 1
        return sucsess_count

    def _devide_imgs_to_clusters(self, img_lst):
        num_of_img_devided = 0
        for img in img_lst:
            index = self._get_closest_cluster_index_using_diff(img)
            self.clusters[index].append(img)

            num_of_img_devided+=1
            if num_of_img_devided%100==0:
                print("num_of_img_devided %s: "%num_of_img_devided)

    def _train_model(self):
        counter = 1
        while -1 in self.cluster_labels and counter < self.MAX_REPS:
            self._devide_imgs_to_clusters(self.img_train)
            self.clusters_rep = [self._get_cluster_new_rep(i) for i in range(self.K)]
            print("modole was trained for %s times"%counter)
            counter+=1

        for index in range(self.K):
            if self.cluster_labels[index] == -1:
                self._fix_cluster_label_to_most_common(index)

    def _get_cluster_new_rep(self,index):
        if self.cluster_labels[index]!=-1:
            return self.clusters_rep[index]
        else:
            rep = self._get_cluster_avg_rep(index)
            if (rep == self.clusters_rep[index]).all():
                self._fix_cluster_label_to_most_common(index)
            return rep

    def _get_cluster_avg_rep(self,index):
        cluster_sum = np.zeros((28, 28)).astype(int)
        for img in self.clusters[index]:
            cluster_sum += img.img
        cluster_avg = (cluster_sum/(len(self.clusters[index])*self.BIT_AVG_THRESHOLD)).astype(int)
        return cluster_avg

    def _fix_cluster_label_to_most_common(self,index):
        lst = [item.label for item in self.clusters[index]]
        mc = self._most_common(lst)
        self.cluster_labels[index] = mc
        print("label of cluster %s was fixed to %s"%(index,mc))

    def _most_common(self,lst):
        if len(lst) == 0:
            return -1
        data = Counter(lst)
        return data.most_common(1)[0][0]

    def _get_closest_cluster_index(self, img):
        i = 0
        max_similarity_index = 0
        max_similarity = -1
        for rep in self.clusters_rep:
            curr_similarity = self._similarity(rep, img.img)
            if curr_similarity > max_similarity:
                max_similarity_index = i
                max_similarity = curr_similarity
            i += 1
        return max_similarity_index

    def _get_closest_cluster_index_using_diff(self, img):
        i = 0
        min_diff_index = 0
        min_diff = self.MAX_INT
        for rep in self.clusters_rep:
            curr_diff = self._diff(rep, img.img)
            if curr_diff < min_diff:
                min_diff_index = i
                min_diff = curr_diff
            i += 1
        return min_diff_index

    def _diff(self, img1, img2):
        '''

        :param img1:
        :param img2:
        :return(int): a number thar represent how "close" img1 and img2 are
        '''
        sum_diff = 0
        for i in list(range(len(img1))):
            for j in range(len(img1[i])):
                img1_pixel = self._get_pix_value(img1[i][j])
                img2_pixel = self._get_pix_value(img2[i][j])
                sum_diff += abs(img1_pixel - img2_pixel)
        return sum_diff

    def _get_pix_value(self,pix):
        if pix<1:
            return 0
        return 1

    def _similarity(self, img1, img2):
        '''

        :param img1:
        :param img2:
        :return(int): a number thar represent how "close" img1 and img2 are
        '''
        sum = 0
        for i in list(range(len(img1))):
            for j in range(len(img1[i])):
                img1_pixel = int((img1[i][j]))
                img2_pixel = int((img2[i][j]))
                sum += abs(img1_pixel * img2_pixel)
        return sum

    def load_data(self):
        mnist_dataLoader = MnistDataloader(MnistDataloader.TRAINING_IMAGES_FILEPATH,
                                           MnistDataloader.TRAINING_LABELS_FILEPATH,
                                           MnistDataloader.TEST_IMAGES_FILEPATH, MnistDataloader.TEST_LABELS_FILEPATH)
        (img_train, label_train), (img_test, label_test) = mnist_dataLoader.load_data()
        self.img_train = [Image(tup[0],tup[1]) for tup in list(zip(img_train, label_train))][:10000]
        self.img_test = [Image(tup[0],tup[1]) for tup in list(zip(img_test, label_test))]

class Image():
    def __init__(self,img,label):
        self.img = (np.array(img)/Ex3.BIT_SIZE).astype(int)
        self.label = label

if __name__ == "__main__":
    ex3 = Ex3()
    ex3.run()
    print("done")
