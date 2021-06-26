from naivebayes import NaiveBayes


class Main:
    @staticmethod
    def main():
        nb = NaiveBayes()
        nb.load_data_training()
        nb.mulai_training()
        nilaiUmur = input("Masukkan nilai Umur[Tua/Muda] : ")
        nilaiStatus = input("Masukkan nilai Status [Belum Kawin/Kawin/Cerai]: ")
        nilaiPendidikan = input("Masukkan nilai Pendidikan [Tidak Sekolah/SD/SLTP/SLTA/Sarjana] : ")
        nilaiTanggungan = input("Masukkan nilai Tanggungan [Tidak Ada/1/2/3/4/5]: ")
        nilaiPekerjaan = input("Masukkan nilai Pekerjaan [Tiada/Buruh Lepas/Petani/Aparatur Negara] : ")
        nilaiPenghasilan = input("Masukkan nilai Penghasilan [Tiada/Rendah/Sedang/Tinggi] : ")


        hasil_prediksi = nb.prediksi(nilai_umur=nilaiUmur,
                                     nilai_status=nilaiStatus,
                                     nilai_pendidikan=nilaiPendidikan,
                                     nilai_tanggungan=nilaiTanggungan,
                                     nilai_pekerjaan=nilaiPekerjaan,
                                     nilai_penghasilan=nilaiPenghasilan)
        print('=====================================')

        print('Hasil akhir prediksi = {}, dengan peluang sebesar {}%'.format(hasil_prediksi['hasil'],
                                                                             hasil_prediksi['peluang']))


Main.main()
