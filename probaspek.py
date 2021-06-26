class ProbAspek:

    def __init__(self, nama_aspek: str, nilai_aspek: str):
        pass
    # TODO: [LANGKAH-1] Buat class untuk menampung nilai matriks probabilitas

        self.nama_aspek = nama_aspek
        self.nilai_aspek = nilai_aspek
        self.jml_miskin = 0
        self.jml_tidak_miskin = 0
        self.p_aspek_miskin = 0
        self.p_aspek_tidak_miskin = 0
    #
    def hitung_p_aspek_miskin(self, jml_total_miskin_aspek):
        # try:
        self.p_aspek_miskin = self.jml_miskin / jml_total_miskin_aspek
        # except ZeroDivisionError:
        #     self.p_aspek_miskin = 0

        return self
    #
    def hitung_p_aspek_tidak_miskin(self, jml_total_tidak_miskin_aspek):
        # try:
        self.p_aspek_tidak_miskin = self.jml_tidak_miskin / jml_total_tidak_miskin_aspek
        # except ZeroDivisionError:
        #     self.p_aspek_tidak_miskin = 0

        return self
    #
    def print(self):
        print('Aspek    : {}'.format(self.nama_aspek))
        print('Nilai    : {}'.format(self.nilai_aspek))
        print('Jml Miskin: {}'.format(self.jml_miskin))
        print('Jml Tidak Miskin: {}'.format(self.jml_tidak_miskin))
        print('P({}|Miskin): {}'.format(self.nilai_aspek, self.p_aspek_miskin))
        print('P({}|Tidak Miskin): {}'.format(self.nilai_aspek, self.p_aspek_tidak_miskin))
        print('------------------------------------------')
    #
    @staticmethod
    def hitung_jml_total_aspek(pa_list: list) -> dict:
        jumlah = {'Miskin': 0, 'Tidak Miskin': 0}
        for pa in pa_list:
            jumlah['Miskin'] += pa.jml_miskin
            jumlah['Tidak Miskin'] += pa.jml_tidak_miskin
        return jumlah
    #
    @staticmethod
    def print_matrix_probabilitas(pa_list: list):
        for pa in pa_list:
            pa.print()
