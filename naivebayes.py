import pandas as pd
from probaspek import ProbAspek


class NaiveBayes:

    def __init__(self):
        pass

        # TODO: [LANGKAH-2] Buat property untuk menampung data dari file CSV
        self.data_training = None

        # TODO: [Langkah-3] Buat variabel dictionary untuk menampung matriks Probabilitas untuk semua aspek
        self.aspek_umur = {'Tua': None, 'Muda': None}
        self.aspek_status = {'Belum Kawin': None, 'Kawin': None, 'Cerai': None}
        self.aspek_pendidikan = {'Tidak Sekolah': None, 'SD': None, 'SLTP': None, 'SLTA': None, 'Sarjana': None}
        self.aspek_tanggungan = {'0': None, '1': None, '2': None, '3': None, '4': None, '5': None}
        self.aspek_pekerjaan = {'Tiada': None, 'Buruh Lepas': None, 'Petani': None, 'Aparatur Negara': None}
        self.aspek_penghasilan = {'Tiada': None, 'Rendah': None, 'Sedang': None, 'Tinggi': None}

        # TODO: [Langkah-4] Buat variabel untuk menampung Prior Probability
        self.prior_probability = {'Miskin': 0, 'Tidak Miskin': 0}

    # TODO: [LANGKAH-5] Load data training dari file CSV
    def load_data_training(self):
        self.data_training = pd.read_csv('daftar_masyarakat_miskin.csv', sep=';')
        print(self.data_training)
        print('-------------------------------------------------------------')

    # TODO: [LANGKAH-6] Membuat object ProbAspek untuk semua nilai pada aspek, sekaligus menghitung jumlah miskin dan tidak miskin
    def buat_prob_aspek(self, nama_aspek: str, nilai_aspek: str) -> ProbAspek:
        #probilitas asek kosong
        prob_aspek = ProbAspek(nama_aspek, nilai_aspek)
        #menghitung jumlah miskin dan tidak miskin
        tua_miskin = self.data_training.loc[(self.data_training[nama_aspek] == nilai_aspek) &
                                             (self.data_training['Actual Class'] == 'Miskin')]
        tua_tidak_miskin = self.data_training.loc[(self.data_training[nama_aspek] == nilai_aspek) &
                                             (self.data_training['Actual Class'] == 'Tidak Miskin')]
        prob_aspek.jml_miskin = len(tua_miskin)
        prob_aspek.jml_tidak_miskin= len(tua_tidak_miskin)
        return prob_aspek

    # TODO: [LANGKAH-7] Mengisi semua nilai pada matris probabilitas aspek
    def mulai_training(self):
        # Aspek Umur
        pu_tua = self.buat_prob_aspek('Umur', 'Tua')
        pu_muda = self.buat_prob_aspek('Umur', 'Muda')

        # Jadikan array
        arr_pu = [pu_tua, pu_muda]
        # Hitung total masing-masing nilai aspek berapa kali muncul di miskin dan tidak miskin
        total_u = ProbAspek.hitung_jml_total_aspek(arr_pu)

        # Hitung probabilitas aspek untuk masing-masing nilai aspek
        pu_tua.hitung_p_aspek_miskin(total_u['Miskin']).hitung_p_aspek_tidak_miskin(total_u['Tidak Miskin'])
        pu_muda.hitung_p_aspek_miskin(total_u['Miskin']).hitung_p_aspek_tidak_miskin(total_u['Tidak Miskin'])

        # Print matrix probabilitas, tetapi bentuknya vertikal, bukan tabel
        ProbAspek.print_matrix_probabilitas(arr_pu)
        self.aspek_umur['Tua'] = pu_tua
        self.aspek_umur['Muda'] = pu_muda


        # TODO :1.1 Status
        # Aspek Status
        ps_belum_kawin = self.buat_prob_aspek('Status', 'Belum Kawin')
        ps_kawin = self.buat_prob_aspek('Status', 'Kawin')
        ps_cerai = self.buat_prob_aspek('Status', 'Cerai')
        # Jadikan array
        arr_ps = [ps_belum_kawin, ps_kawin, ps_cerai]
        # Hitung total masing-masing nilai aspek berapa kali muncul di miskin dan tidak miskin
        total_s = ProbAspek.hitung_jml_total_aspek(arr_ps)
        # Hitung probabilitas aspek untuk masing-masing nilai aspek
        ps_belum_kawin.hitung_p_aspek_miskin(total_s['Miskin']).hitung_p_aspek_tidak_miskin(total_s['Tidak Miskin'])
        ps_kawin.hitung_p_aspek_miskin(total_s['Miskin']).hitung_p_aspek_tidak_miskin(total_s['Tidak Miskin'])
        ps_cerai.hitung_p_aspek_miskin(total_s['Miskin']).hitung_p_aspek_tidak_miskin(total_s['Tidak Miskin'])

        # Print matrix probabilitas, tetapi bentuknya vertikal, bukan tabel
        ProbAspek.print_matrix_probabilitas(arr_ps)
        self.aspek_status['Belum Kawin'] = ps_belum_kawin
        self.aspek_status['Kawin'] = ps_kawin
        self.aspek_status['Cerai'] = ps_cerai

        # TODO :1.2 Pendidikan
        # Aspek Pendidikan
        pn_tidak_sekolah = self.buat_prob_aspek('Pendidikan', 'Tidak Sekolah')
        pn_sd = self.buat_prob_aspek('Pendidikan', 'SD')
        pn_sltp = self.buat_prob_aspek('Pendidikan', 'SLTP')
        pn_slta = self.buat_prob_aspek('Pendidikan', 'SLTA')
        pn_sarjana = self.buat_prob_aspek('Pendidikan', 'Sarjana')

        # Jadikan array
        arr_pn = [pn_tidak_sekolah, pn_sd, pn_sltp, pn_slta, pn_sarjana]
        # Hitung total masing-masing nilai aspek berapa kali muncul di miskin dan tidak miskin
        total_pn = ProbAspek.hitung_jml_total_aspek(arr_pn)
        # Hitung probabilitas aspek untuk masing-masing nilai aspek
        pn_tidak_sekolah.hitung_p_aspek_miskin(total_pn['Miskin']).hitung_p_aspek_tidak_miskin(total_pn['Tidak Miskin'])
        pn_sd.hitung_p_aspek_miskin(total_pn['Miskin']).hitung_p_aspek_tidak_miskin(total_pn['Tidak Miskin'])
        pn_sltp.hitung_p_aspek_miskin(total_pn['Miskin']).hitung_p_aspek_tidak_miskin(total_pn['Tidak Miskin'])
        pn_slta.hitung_p_aspek_miskin(total_pn['Miskin']).hitung_p_aspek_tidak_miskin(total_pn['Tidak Miskin'])
        pn_sarjana.hitung_p_aspek_miskin(total_pn['Miskin']).hitung_p_aspek_tidak_miskin(total_pn['Tidak Miskin'])

        # Print matrix probabilitas, tetapi bentuknya vertikal, bukan tabel
        ProbAspek.print_matrix_probabilitas(arr_pn)
        self.aspek_pendidikan['Tidak Sekolah'] = pn_tidak_sekolah
        self.aspek_pendidikan['SD'] = pn_sd
        self.aspek_pendidikan['SLTP'] = pn_sltp
        self.aspek_pendidikan['SLTA'] = pn_slta
        self.aspek_pendidikan['Sarjana'] = pn_sarjana

        # TODO :1.3 Tanggungan
        # Aspek Tanggungan
        pt_nol = self.buat_prob_aspek('Tanggungan', 'Tidak Ada')
        pt_satu = self.buat_prob_aspek('Tanggungan', '1')
        pt_dua = self.buat_prob_aspek('Tanggungan', '2')
        pt_tiga = self.buat_prob_aspek('Tanggungan', '3')
        pt_empat = self.buat_prob_aspek('Tanggungan', '4')
        pt_lima = self.buat_prob_aspek('Tanggungan', '5')

        # Jadikan array
        arr_pt = [pt_nol, pt_satu, pt_dua, pt_tiga, pt_empat, pt_lima]
        # Hitung total masing-masing nilai aspek berapa kali muncul di miskin dan tidak miskin
        total_pt = ProbAspek.hitung_jml_total_aspek(arr_pt)
        # Hitung probabilitas aspek untuk masing-masing nilai aspek
        pt_nol.hitung_p_aspek_miskin(total_pt['Miskin']).hitung_p_aspek_tidak_miskin(total_pt['Tidak Miskin'])
        pt_satu.hitung_p_aspek_miskin(total_pt['Miskin']).hitung_p_aspek_tidak_miskin(total_pt['Tidak Miskin'])
        pt_dua.hitung_p_aspek_miskin(total_pt['Miskin']).hitung_p_aspek_tidak_miskin(total_pt['Tidak Miskin'])
        pt_tiga.hitung_p_aspek_miskin(total_pt['Miskin']).hitung_p_aspek_tidak_miskin(total_pt['Tidak Miskin'])
        pt_empat.hitung_p_aspek_miskin(total_pt['Miskin']).hitung_p_aspek_tidak_miskin(total_pt['Tidak Miskin'])
        pt_lima.hitung_p_aspek_miskin(total_pt['Miskin']).hitung_p_aspek_tidak_miskin(total_pt['Tidak Miskin'])

        # Print matrix probabilitas, tetapi bentuknya vertikal, bukan tabel
        ProbAspek.print_matrix_probabilitas(arr_pt)
        self.aspek_tanggungan['Tidak Ada'] = pt_nol
        self.aspek_tanggungan['1'] = pt_satu
        self.aspek_tanggungan['2'] = pt_dua
        self.aspek_tanggungan['3'] = pt_tiga
        self.aspek_tanggungan['4'] = pt_empat
        self.aspek_tanggungan['5'] = pt_lima

        # TODO :1.4 Pekerjaan
        # Aspek Pekerjaan
        pk_tiada = self.buat_prob_aspek('Pekerjaan', 'Tiada')
        pk_buruh = self.buat_prob_aspek('Pekerjaan', 'Buruh Lepas')
        pk_petani = self.buat_prob_aspek('Pekerjaan', 'Petani')
        pk_aparatur = self.buat_prob_aspek('Pekerjaan', 'Aparatur Negara')


        # Jadikan array
        arr_pk = [pk_tiada, pk_buruh, pk_petani, pk_aparatur]
        # Hitung total masing-masing nilai aspek berapa kali muncul di miskin dan tidak miskin
        total_pk = ProbAspek.hitung_jml_total_aspek(arr_pk)
        # Hitung probabilitas aspek untuk masing-masing nilai aspek
        pk_tiada.hitung_p_aspek_miskin(total_pk['Miskin']).hitung_p_aspek_tidak_miskin(total_pk['Tidak Miskin'])
        pk_buruh.hitung_p_aspek_miskin(total_pk['Miskin']).hitung_p_aspek_tidak_miskin(total_pk['Tidak Miskin'])
        pk_petani.hitung_p_aspek_miskin(total_pk['Miskin']).hitung_p_aspek_tidak_miskin(total_pk['Tidak Miskin'])
        pk_aparatur.hitung_p_aspek_miskin(total_pk['Miskin']).hitung_p_aspek_tidak_miskin(total_pk['Tidak Miskin'])

        # Print matrix probabilitas, tetapi bentuknya vertikal, bukan tabel
        ProbAspek.print_matrix_probabilitas(arr_pk)
        self.aspek_pekerjaan['Tiada'] = pk_tiada
        self.aspek_pekerjaan['Buruh Lepas'] = pk_buruh
        self.aspek_pekerjaan['Petani'] = pk_petani
        self.aspek_pekerjaan['Aparatur Negara'] = pk_aparatur

        # TODO :1.5 Penghasilan
        # Aspek Penghasilan
        ph_tiada = self.buat_prob_aspek('Penghasilan', 'Tiada')
        ph_rendah = self.buat_prob_aspek('Penghasilan', 'Rendah')
        ph_sedang = self.buat_prob_aspek('Penghasilan', 'Sedang')
        ph_tinggi = self.buat_prob_aspek('Penghasilan', 'Tinggi')

        # Jadikan array
        arr_ph = [ph_tiada, ph_rendah, ph_sedang, ph_tinggi]
        # Hitung total masing-masing nilai aspek berapa kali muncul di miskin dan tidak miskin
        total_ph = ProbAspek.hitung_jml_total_aspek(arr_ph)
        # Hitung probabilitas aspek untuk masing-masing nilai aspek
        ph_tiada.hitung_p_aspek_miskin(total_ph['Miskin']).hitung_p_aspek_tidak_miskin(total_ph['Tidak Miskin'])
        ph_rendah.hitung_p_aspek_miskin(total_ph['Miskin']).hitung_p_aspek_tidak_miskin(total_ph['Tidak Miskin'])
        ph_sedang.hitung_p_aspek_miskin(total_ph['Miskin']).hitung_p_aspek_tidak_miskin(total_ph['Tidak Miskin'])
        ph_tinggi.hitung_p_aspek_miskin(total_ph['Miskin']).hitung_p_aspek_tidak_miskin(total_ph['Tidak Miskin'])


        # Print matrix probabilitas, tetapi bentuknya vertikal, bukan tabel
        ProbAspek.print_matrix_probabilitas(arr_ph)
        self.aspek_penghasilan['Tiada'] = ph_tiada
        self.aspek_penghasilan['Rendah'] = ph_rendah
        self.aspek_penghasilan['Sedang'] = ph_sedang
        self.aspek_penghasilan['Tinggi'] = ph_tinggi


    # TODO: [LANGKAH-8] Menghitung prior probability
    def hitung_prior_probability(self):
        pp_miskin = self.buat_prob_aspek('Actual Class', 'Miskin')
        pp_tidak_miskin = self.buat_prob_aspek('Actual Class', 'Tidak Miskin')
        arr_pp = (pp_miskin, pp_tidak_miskin)
        total_pp = ProbAspek.hitung_jml_total_aspek(arr_pp)
        self.prior_probability['Miskin'] = total_pp['Miskin'] / (total_pp['Miskin'] + total_pp['Tidak Miskin'])
        self.prior_probability['Tidak Miskin'] = total_pp['Tidak Miskin'] / (total_pp['Miskin'] + total_pp['Tidak Miskin'])
        # TODO: [SOAL-2] Prior Probability-nya masih 0, hitunglah prior probability yang sebenarnya!

    # TODO: [LANGKAH-9] Membuat method untuk memprediksi hasil akhir berdasarkan nilai aspek
    def prediksi(self, nilai_umur: str, nilai_status: str, nilai_pendidikan: str, nilai_tanggungan: str, nilai_pekerjaan: str, nilai_penghasilan: str):

        self.hitung_prior_probability()
        predict_miskin = self.prior_probability['Miskin'] * \
                        self.aspek_umur[nilai_umur].p_aspek_miskin * \
                        self.aspek_status[nilai_status].p_aspek_miskin * \
                        self.aspek_pendidikan[nilai_pendidikan].p_aspek_miskin * \
                        self.aspek_tanggungan[nilai_tanggungan].p_aspek_miskin * \
                        self.aspek_pekerjaan[nilai_pekerjaan].p_aspek_miskin * \
                        self.aspek_penghasilan[nilai_penghasilan].p_aspek_miskin
        print('Peluang Miskin: {}'.format(predict_miskin))

        predict_tidak_miskin = self.prior_probability['Tidak Miskin'] * \
                        self.aspek_umur[nilai_umur].p_aspek_tidak_miskin * \
                        self.aspek_status[nilai_status].p_aspek_tidak_miskin * \
                        self.aspek_pendidikan[nilai_pendidikan].p_aspek_tidak_miskin * \
                        self.aspek_tanggungan[nilai_tanggungan].p_aspek_tidak_miskin * \
                        self.aspek_pekerjaan[nilai_pekerjaan].p_aspek_tidak_miskin * \
                        self.aspek_penghasilan[nilai_penghasilan].p_aspek_tidak_miskin
        print('Peluang Tidak Miskin: {}'.format(predict_tidak_miskin))


        if predict_tidak_miskin > predict_miskin:
            hasil = "Tidak Miskin"
            peluang = predict_tidak_miskin
        else:
            hasil = "Miskin"
            peluang = predict_miskin
        return {'hasil': hasil, 'peluang': peluang}
