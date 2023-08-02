from tkinter import *  
from tkinter import filedialog, simpledialog, messagebox, ttk, IntVar  
from PIL import Image, ImageTk, ImageOps, ImageFilter 
import random 
import cv2  
import numpy as np  
import matplotlib.pyplot as plt
from collections import Counter  
import sys 
from tkinter.colorchooser import askcolor
from tkinter import Toplevel
import huffman
from skimage import segmentation, color, morphology

class ProcessadorImagens:
    def __init__(self):
        # Cria uma janela para o processamento de imagens
        self.janela = Tk()
        self.janela.title("PDI - Processamento Digital de Imagens")
        self.image = None
        self.original_image = None  
        self.image_path = ""
        self.var_ajustada = IntVar(value=0)  

        
        # Botão para desfazer a última ação
        self.botao_desfazer = Button(self.janela, text="Desfazer", command=self.desfazer)
        self.botao_desfazer.pack(side=TOP)

        # Botão para sair da aplicação
        self.botao_sair = Button(self.janela, text="Sair", command=self.sair)
        self.botao_sair.pack(side=BOTTOM)


        # Criação do menu da janela
        menu = Menu(self.janela)
        self.janela.config(menu=menu)

        # Menu "File" para carregar e salvar imagens
        file_menu = Menu(menu)
        menu.add_cascade(label="Carregar imagem", menu=file_menu)
        file_menu.add_command(label="Abrir", command=self.load_imagem)
        file_menu.add_command(label="Salvar", command=self.save_imagem)

        # Criação do canvas para exibir a imagem
        self.canvas = Canvas(self.janela, bg='gray')
        self.canvas.pack(fill='both', expand=True)

        # Criar um rótulo para exibir o template
        self.label_template = ttk.Label(self.janela)
        self.label_template.pack(side=LEFT)
        
        # Menu "Processamento" com as opções de processamento de imagens
        imagem_menu = Menu(menu)
        menu.add_cascade(label="Processamento-parte 01", menu=imagem_menu)
        imagem_menu.add_command(label="Converter em Escala de Cinza", command=self.converte_escala_cinza)
        imagem_menu.add_command(label="Transformação de Potência", command=self.transformacao_potencia)
        imagem_menu.add_command(label="Alargamento de Contraste", command=self.alargamento_contraste)
        imagem_menu.add_command(label="Fatiamento por Planos de Bits", command=self.fatiamento_planos_bits)
        imagem_menu.add_command(label="Equalização de Histograma", command=self.show_histogram_image)
        imagem_menu.add_command(label="Fatiamento por Intensidades para Pseudocores", command=self.fatiamento_pseudocores)
        imagem_menu.add_command(label="Máximo", command=lambda: self.aplicar_filtro_menu("Máximo"))
        imagem_menu.add_command(label="Mínimo", command=lambda: self.aplicar_filtro_menu("Mínimo"))
        imagem_menu.add_command(label="Média", command=lambda: self.aplicar_filtro_menu("Média"))
        imagem_menu.add_command(label="Mediana", command=lambda: self.aplicar_filtro_menu("Mediana"))
        imagem_menu.add_command(label="Laplaciano", command=lambda: self.aplicar_filtro_menu("Laplaciano"))

        imagenew_menu = Menu(menu)
        menu.add_cascade(label="Processamento-parte 02", menu=imagenew_menu)
        imagenew_menu.add_command(label="Computar DFT e IDFT", command=self.dft_idft)
        imagenew_menu.add_command(label="Computar Espectro da DFT", command=self.espectro_dft)
        imagenew_menu.add_command(label="Computar Ângulo de Fase", command=self.angulo_fase)
        imagenew_menu.add_command(label="Filtro Passa-Baixa Ideal", command=self.filtro_passa_baixa)
        imagenew_menu.add_command(label="Filtro Passa-Alta Ideal", command=self.filtro_passa_alta)
        imagenew_menu.add_command(label="Filtro Rejeita-notch Ideal ", command=self.filtro_rejeita_notch)
        imagenew_menu.add_command(label="Gerar Ruído Gaussiano ", command=self.ruido_gaussiano)
        imagenew_menu.add_command(label="Gerar Ruído Sal e Pimenta ", command=self.ruido_sal_pimenta)
        imagenew_menu.add_command(label="Filtro da Média Geométrica", command=self.filtro_media_geometrica)
        imagenew_menu.add_command(label="Filtro da Média Alfa Cortada",command=lambda: self.filtro_media_alfa_cortada(5))
        imagenew_menu.add_command(label="Processamento Morfológico", command=self.processamento_morfologico)
        imagenew_menu.add_command(label="Codificacao_Huffman", command=self.codificacao_huffman)

        self.raio_var = IntVar()
        self.raio_var.set(10) 

        imagem3_menu = Menu(menu)
        menu.add_cascade(label="Processamento-parte 03", menu=imagem3_menu)
        imagem3_menu.add_command(label="Detector de bordas do Canny", command=self.deteccao_bordas_canny)
        imagem3_menu.add_command(label="Segmentação por Crescimento de Regiões", command=self.segmentacao_crescimento_regioes)
        imagem3_menu.add_command(label="Algoritmo código da cadeia", command=self.codigo_da_cadeia)
        imagem3_menu.add_command(label="Esqueletizaçao", command=self.esqueletizacao)
        imagem3_menu.add_checkbutton(label='Casamento por correlação', command=self.casamento_por_correlacao)
        
        
        # Configuração da ação ao fechar a janela
        self.janela.protocol("WM_DELETE_WINDOW", self.fechar_janela)
        self.janela.mainloop()

    
    def load_imagem(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.original_image = self.image.copy()
            self.image_path = file_path
            self.show_imagem()
        
    
    def show_imagem(self):
        if self.image:
            img_tk = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, image=img_tk, anchor=NW)
            self.canvas.image = img_tk

    
    def save_imagem(self):
        if self.image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                           filetypes=[("Imagens", "*.png;*.jpg;*.jpeg; *.bmp;*.gif")])
            if file_path:
                self.image.save(file_path)
    
    
    def converte_escala_cinza(self):
        if self.image:
            self.image = self.image.convert('L')
            self.show_imagem()

    
    def transformacao_potencia(self):
        if self.image:
            gama_input = simpledialog.askstring("Transformação de Potência", "Digite o valor de gama:")
            if gama_input is not None:
                try:
                    gama = float(gama_input.replace(',', '.'))
                    img_array = np.array(self.image)
                    img_transformed = np.power(img_array, gama)
                    img_transformed = img_transformed * (255.0 / img_transformed.max())
                    self.image = Image.fromarray(img_transformed.astype(np.uint8))
                    self.show_imagem()
                except ValueError:
                    messagebox.showerror("Erro", "Valor de gama inválido!")

    
    def alargamento_contraste(self):
        if self.image:
            # Solicitar os valores dos pontos de referência através de caixas de diálogo
            r1_input = simpledialog.askstring("Alargamento de Contraste", "Digite o valor de r1:")
            s1_input = simpledialog.askstring("Alargamento de Contraste", "Digite o valor de s1:")
            r2_input = simpledialog.askstring("Alargamento de Contraste", "Digite o valor de r2:")
            s2_input = simpledialog.askstring("Alargamento de Contraste", "Digite o valor de s2:")

            if r1_input is not None and s1_input is not None and r2_input is not None and s2_input is not None:
                try:
                    # Converter os valores de entrada para inteiros
                    r1 = int(r1_input)
                    s1 = int(s1_input)
                    r2 = int(r2_input)
                    s2 = int(s2_input)
    
                    a1 = 255 / (r2 - r1)
                    a2 = 255 / (s2 - s1)
                    a3 = (255 - s2) / (255 - r2)
                    self.image = self.image.point(lambda x: int(a1 * x) if x < r1 else int(a2 * (x - r1) + s1) if r1 <= x < r2 else int(a3 * (x - r2) + s2))
                    self.show_imagem()
                except ValueError:
                    messagebox.showerror("Erro", "Valores inválidos!")


    def fatiamento_planos_bits(self):
        if self.image:
            plano_bits = simpledialog.askinteger("Fatiamento por Planos de Bits",
                                                 "Digite o número do plano de bits (0-7): ", minvalue=0, maxvalue=7)
            if plano_bits is not None:
                self.image = self.image.convert('L')
                pixels = self.image.load()
                width, height = self.image.size
                for x in range(width):
                    for y in range(height):
                        pixel = pixels[x, y]
                        novo_pixel = (pixel >> plano_bits) & 1
                        novo_pixel = novo_pixel * 255
                        pixels[x, y] = novo_pixel
                self.show_imagem()

    
    def show_histogram_image(self):
        if self.image_path:
            img = cv2.imread(self.image_path, 0)
            hist, bins = np.histogram(img.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            plt.subplot(121), plt.imshow(img, 'gray')
            plt.title('Imagem'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.plot(cdf_normalized, color='b')
            plt.hist(img.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
            plt.legend(('cdf', 'histograma'), loc='upper left')
            plt.title('Histograma')
            plt.show()

    
    def fatiamento_pseudocores(self):
        if self.image:
            paleta_cores = [
                (0, 0, 0),      # Pseudocor 0: Preto
                (255, 0, 0),    # Pseudocor 1: Vermelho
                (0, 255, 0),    # Pseudocor 2: Verde
                (0, 0, 255),    # Pseudocor 3: Azul
                (255, 255, 255),    # Pseudocor 4: Branco
                (255, 255, 0),  # Pseudocor 5: Amarelo
                (255, 0, 255),  # Pseudocor 6: Magenta
                (0, 255, 255)   # Pseudocor 7: Ciano
            ]

            self.image = self.image.convert('L')
            width, height = self.image.size

            # Determinar a amplitude de cada faixa
            amplitude = 256 // len(paleta_cores)

            # Criar uma nova imagem para armazenar os pseudocores
            nova_imagem = Image.new('RGB', (width, height))

            for x in range(width):
                for y in range(height):
                    pixel = self.image.getpixel((x, y))
                    faixa = pixel // amplitude
                    novo_pixel = paleta_cores[faixa]
                    nova_imagem.putpixel((x, y), novo_pixel)

            self.image = nova_imagem
            self.show_imagem()

    
    def aplicar_filtro_menu(self, filtro):
        # Verifica se há uma imagem carregada
        if self.image:
            # Verifica qual filtro foi selecionado
            if filtro == "Máximo":
                self.image = self.image.filter(ImageFilter.MaxFilter)
            elif filtro == "Mínimo":
                self.image = self.image.filter(ImageFilter.MinFilter)
            elif filtro == "Média":
                self.image = self.image.filter(ImageFilter.MedianFilter)
            elif filtro == "Mediana":
                self.image = self.image.filter(ImageFilter.MedianFilter)
            elif filtro == "Laplaciano":
                self.image = self.laplacian_filter()
            self.show_imagem()

    
    def laplacian_filter(self):
        # Carrega a imagem usando o OpenCV em escala de cinza
        if self.image:
            img = cv2.imread(self.image_path, 0)
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            laplacian_img = Image.fromarray(laplacian)
            return laplacian_img
        

    def dft_idft(self):
        if self.image:
            gray_image = self.image.convert('L')
            np_image = np.array(gray_image)
            dft = np.fft.fft2(np_image)
            idft = np.fft.ifft2(dft)
            idft = idft.real
            idft = (idft - np.min(idft)) * (255 / (np.max(idft) - np.min(idft)))
            idft_image = Image.fromarray(idft.astype(np.uint8))

            plt.subplot(1, 2, 1)
            plt.imshow(np_image, cmap='gray')
            plt.title('Imagem original')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(idft_image, cmap='gray')
            plt.title('IDFT')
            plt.axis('off')

            plt.show()

    def espectro_dft(self):
        if self.image:

            gray_image = self.image.convert('L')
            dft = np.fft.fft2(gray_image)
            dft_shift = np.fft.fftshift(dft)
            espectro = 20 * np.log(np.abs(dft_shift))

        plt.subplot(1, 2, 1)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Imagem original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(espectro, cmap='gray')
        plt.title('Espectro de Magnitude (DFT)')
        plt.axis('off')

        plt.show()

    def angulo_fase(self):
        if self.image:
            gray_image = self.image.convert('L')
            dft = np.fft.fft2(gray_image)
            dft_shift = np.fft.fftshift(dft)
            angulo_fase = np.angle(dft_shift)

        plt.subplot(1, 2, 1)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Imagem original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(angulo_fase, cmap='gray')
        plt.title('Ângulo de fase ')
        plt.axis('off')

        plt.show()

    def filtro_passa_baixa(self):
        if self.image:
            gray_image = self.image.convert('L')
            width, height = gray_image.size
            np_image = np.array(gray_image)
            dft = np.fft.fft2(np_image)
            dft_shift = np.fft.fftshift(dft)
            raio = simpledialog.askinteger("Raio do Filtro", "Insira o tamanho do raio para o filtro passa-baixa:")

            if raio:
                filtro = np.zeros_like(dft_shift)
                filtro[(height // 2) - raio: (height // 2) + raio, (width // 2) - raio: (width // 2) + raio] = 1
                dft_filtered = dft_shift * filtro
                idft = np.fft.ifftshift(dft_filtered)
                idft = np.fft.ifft2(idft)
                idft = idft.real
                idft = (idft - np.min(idft)) * (255 / (np.max(idft) - np.min(idft)))
                idft_image = Image.fromarray(idft.astype(np.uint8))

                plt.subplot(1, 2, 1)
                plt.imshow(np_image, cmap='gray')
                plt.title('Imagem original')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(idft_image, cmap='gray')
                plt.title('Filtro Passa-Baixa Ideal')
                plt.axis('off')

                plt.show()

    def filtro_passa_alta(self):
        if self.image:
            gray_image = self.image.convert('L')
            width, height = gray_image.size
            dft = np.fft.fft2(gray_image)
            dft_shift = np.fft.fftshift(dft)
            raio = simpledialog.askinteger("Raio do Filtro", "Insira o tamanho do raio para o filtro passa-alta:")

            if raio:
                filtro = np.ones_like(dft_shift)
                centro_x = width // 2
                centro_y = height // 2
                filtro[centro_y - raio: centro_y + raio, centro_x - raio: centro_x + raio] = 0
                dft_filtered = dft_shift * filtro
                idft = np.fft.ifftshift(dft_filtered)
                idft = np.fft.ifft2(idft)
                idft = idft.real
                idft = np.abs(idft)
                idft[idft > 255] = 255
                idft[idft < 0] = 0
                idft = idft.astype(np.uint8)
                idft_image = Image.fromarray(idft)

                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(gray_image, cmap='gray')
                axs[0].set_title('Imagem original')
                axs[0].axis('off')

                axs[1].imshow(idft_image, cmap='gray')
                axs[1].set_title('Filtro Passa-Alta Ideal')
                axs[1].axis('off')

                plt.show()

    def filtro_rejeita_notch(self):
        if self.image:
            gray_image = self.image.convert('L')
            np_image = np.array(gray_image)
            radius = simpledialog.askinteger("Filtro Rejeita-Notch Ideal", "Digite o tamanho do raio:")
            if radius is not None:
                center_positions = self.selecionar_centros(np_image)
                filtered_image = self.filtro_rejeita_notch_ideal(np_image, center_positions, radius)
                self.image = filtered_image
                self.show_imagem()

    def selecionar_centros(self, image):
        plt.imshow(image, cmap='gray')
        plt.title('Selecione os centros dos filtros')
        plt.axis('off')
        center_positions = []

        def selecionar_ponto(event):
            if event.button == 1 and event.xdata is not None and event.ydata is not None:
                x = int(event.xdata)
                y = int(event.ydata)
                center_positions.append((x, y))
                plt.plot(x, y, 'ro')
                plt.draw()

        plt.connect('button_press_event', selecionar_ponto)
        plt.show()

        return center_positions

    def filtro_rejeita_notch_ideal(self, image, center_positions, radius):
        np_image = np.array(image)
        dft = np.fft.fft2(np_image)
        dft_shift = np.fft.fftshift(dft)

        height, width = np_image.shape
        y, x = np.indices((height, width))
        dist = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)

        mask = np.ones_like(dft_shift)
        for center in center_positions:
            mask[np.logical_and(dist > center[0] - radius, dist < center[0] + radius)] = 0
            mask[np.logical_and(dist > center[1] - radius, dist < center[1] + radius)] = 0

        dft_filtered = dft_shift * mask
        idft = np.fft.ifftshift(dft_filtered)
        idft = np.fft.ifft2(idft)
        idft = idft.real

        idft = np.abs(idft)
        idft[idft > 255] = 255
        idft[idft < 0] = 0
        idft = idft.astype(np.uint8)

        filtered_image = Image.fromarray(idft)

        return filtered_image


    def ruido_gaussiano(self):
        if self.image:
            gray_image = self.image.convert('L')
            np_image = np.array(gray_image)
            height, width = np_image.shape
            desvio_padrao = simpledialog.askfloat("Ruído Gaussiano", "Insira o desvio padrão do ruído gaussiano:")

            if desvio_padrao:
                ruido = np.random.normal(0, desvio_padrao, (height, width))
                imagem_ruidosa = np_image + ruido
                imagem_ruidosa = np.clip(imagem_ruidosa, 0, 255).astype(np.uint8)
                imagem_ruidosa = Image.fromarray(imagem_ruidosa)

                plt.subplot(1, 2, 1)
                plt.imshow(np_image, cmap='gray')
                plt.title('Imagem original')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(imagem_ruidosa, cmap='gray')
                plt.title('Imagem com Ruído Gaussiano')
                plt.axis('off')

                plt.show()

    def ruido_sal_pimenta(self):
        if self.image:
            gray_image = self.image.convert('L')
            np_image = np.array(gray_image)
            height, width = np_image.shape
            porcentagem = simpledialog.askinteger("Ruído Sal e Pimenta", "Insira a porcentagem de ruído de sal e pimenta:")

            if porcentagem:
                proporcao = porcentagem / 100.0

                num_pixels = int(width * height * proporcao)
                pixels_ruidosos = random.sample(range(width * height), num_pixels)

                imagem_ruidosa = np_image.copy()

                for pixel in pixels_ruidosos:
                    y = pixel // width
                    x = pixel % width
                    if random.random() < 0.5:
                        imagem_ruidosa[y, x] = 255
                    else:
                        imagem_ruidosa[y, x] = 0

                imagem_ruidosa = Image.fromarray(imagem_ruidosa)

                plt.subplot(1, 2, 1)
                plt.imshow(np_image, cmap='gray')
                plt.title('Imagem original')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(imagem_ruidosa, cmap='gray')
                plt.title('Imagem com Ruído de Sal e Pimenta')
                plt.axis('off')

                plt.show()

    def filtro_media_geometrica(self, tamanho_janela=3):
        if self.image:
            gray_image = self.image.convert('L')
            width, height = gray_image.size
            np_image = np.array(gray_image)
            np_image_normalized = np_image / 255.0
            imagem_filtrada = np_image_normalized.copy()
            pixels_cortados = tamanho_janela // 2

            for y in range(pixels_cortados, height - pixels_cortados):
                for x in range(pixels_cortados, width - pixels_cortados):
                    janela = np_image_normalized[y - pixels_cortados: y + pixels_cortados + 1,
                             x - pixels_cortados: x + pixels_cortados + 1]

                    media_geometrica = np.prod(janela) ** (1 / (tamanho_janela ** 2))

                    imagem_filtrada[y, x] = media_geometrica

            imagem_filtrada_denormalized = imagem_filtrada * 255.0

            imagem_filtrada_pil = Image.fromarray(imagem_filtrada_denormalized.astype(np.uint8))

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(np_image, cmap='gray')
            axs[0].set_title('Imagem original')
            axs[0].axis('off')

            axs[1].imshow(imagem_filtrada_pil, cmap='gray')
            axs[1].set_title('Filtro da Média Geométrica')
            axs[1].axis('off')

            plt.show()
    def filtro_media_alfa_cortada(self, tamanho_janela):
        if self.image:
            gray_image = self.image.convert('L')
            width, height = gray_image.size
            np_image = np.array(gray_image)
            imagem_filtrada = np_image.copy()
            pixels_cortados = int(tamanho_janela / 2)

            for y in range(pixels_cortados, height - pixels_cortados):
                for x in range(pixels_cortados, width - pixels_cortados):
                    # Obter a janela ao redor do pixel
                    janela = np_image[y - pixels_cortados: y + pixels_cortados + 1,
                             x - pixels_cortados: x + pixels_cortados + 1]

                    media = np.mean(janela)

                    imagem_filtrada[y, x] = media

            imagem_filtrada = Image.fromarray(imagem_filtrada.astype(np.uint8))

            plt.subplot(1, 2, 1)
            plt.imshow(np_image, cmap='gray')
            plt.title('Imagem original')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(imagem_filtrada, cmap='gray')
            plt.title('Filtro da Média Alfa Cortada')
            plt.axis('off')

            plt.show()


    def processamento_morfologico(self):
        if self.image:
            gray_image = self.image.convert('L')
            np_image = np.array(gray_image)
            elemento_estruturante = self.elemento_estruturante()
            eroded_image = cv2.erode(np_image, elemento_estruturante, iterations=1)
            dilated_image = cv2.dilate(np_image, elemento_estruturante, iterations=1)
            opened_image = cv2.morphologyEx(np_image, cv2.MORPH_OPEN, elemento_estruturante)
            closed_image = cv2.morphologyEx(np_image, cv2.MORPH_CLOSE, elemento_estruturante)

            eroded_image = Image.fromarray(eroded_image.astype(np.uint8))
            dilated_image = Image.fromarray(dilated_image.astype(np.uint8))
            opened_image = Image.fromarray(opened_image.astype(np.uint8))
            closed_image = Image.fromarray(closed_image.astype(np.uint8))

            plt.subplot(2, 2, 1)
            plt.imshow(opened_image, cmap='gray')
            plt.title('Abertura')
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.imshow(dilated_image, cmap='gray')
            plt.title('Dilatação')
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.imshow(eroded_image, cmap='gray')
            plt.title('Erosão')
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.imshow(closed_image, cmap='gray')
            plt.title('Fechamento')
            plt.axis('off')

            plt.show()

    def elemento_estruturante(self):
        tamanho = self.raio_var.get() * 2 + 1
        elemento_estruturante = np.random.randint(0, 2, size=(tamanho, tamanho), dtype=np.uint8)
        return elemento_estruturante
    

    def codificacao_huffman(self):
            if self.image:
                gray_image = self.image.convert('L')
                pixel_counts = Counter(gray_image.getdata())
                total_pixels = gray_image.size[0] * gray_image.size[1]

                probabilidades = {intensidade: count / total_pixels for intensidade, count in pixel_counts.items()}

                probabilidades_ordenadas = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)

                tabela_huffman = {}

                codigo = ''
                for intensidade, probabilidade in probabilidades_ordenadas:
                    tabela_huffman[intensidade] = codigo
                    codigo += '0'

                janela_tabela = Toplevel(self.janela)
                janela_tabela.title("Tabela de Códigos de Huffman")

                tabela_text = Text(janela_tabela)
                tabela_text.pack()

                tabela_text.insert("1.0", "Intensidade\tProbabilidade\tCódigo\n")
                for intensidade, probabilidade in probabilidades_ordenadas:
                    probabilidade_formatada = format(probabilidade, '.4f')
                    codigo = tabela_huffman[intensidade]
                    linha_tabela = f"{intensidade}\t\t{probabilidade_formatada}\t\t{codigo}\n"
                    tabela_text.insert("end", linha_tabela)

    
    def deteccao_bordas_canny(self):
        if self.image:
            gray_image = self.image.convert('L')
            np_image = np.array(gray_image)

            # Aplicar filtro gaussiano para redução de ruído
            blurred_image = cv2.GaussianBlur(np_image, (5, 5), 0)

            # Calcular magnitude do gradiente e ângulos das imagens usando Sobel
            gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            gradient_angles = np.arctan2(gradient_y, gradient_x)

            # Aplicar supressão não máxima
            suppressed_image = np.zeros_like(gradient_magnitude)
            for i in range(1, gradient_magnitude.shape[0] - 1):
                for j in range(1, gradient_magnitude.shape[1] - 1):
                    angle = gradient_angles[i, j] * 180 / np.pi
                    if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (
                            -180 <= angle < -157.5):
                        if (gradient_magnitude[i, j] >= gradient_magnitude[i, j + 1]) and (
                                gradient_magnitude[i, j] >= gradient_magnitude[i, j - 1]):
                            suppressed_image[i, j] = gradient_magnitude[i, j]
                    elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
                        if (gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j + 1]) and (
                                gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j - 1]):
                            suppressed_image[i, j] = gradient_magnitude[i, j]
                    elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
                        if (gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j]) and (
                                gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j]):
                            suppressed_image[i, j] = gradient_magnitude[i, j]
                    else:
                        if (gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j - 1]) and (
                                gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j + 1]):
                            suppressed_image[i, j] = gradient_magnitude[i, j]

            # Limiarização por histerese e análise de conectividade
            threshold_low = 50
            threshold_high = 150
            edges = np.zeros_like(suppressed_image)
            strong_edge_pixels = suppressed_image >= threshold_high
            edges[strong_edge_pixels] = 255
            weak_edge_pixels = (suppressed_image >= threshold_low) & (suppressed_image < threshold_high)
            for i in range(1, edges.shape[0] - 1):
                for j in range(1, edges.shape[1] - 1):
                    if weak_edge_pixels[i, j]:
                        if (edges[i + 1, j] == 255) or (edges[i - 1, j] == 255) or (edges[i, j + 1] == 255) or (
                                edges[i, j - 1] == 255) or (edges[i + 1, j + 1] == 255) or (
                                edges[i - 1, j - 1] == 255) or (edges[i + 1, j - 1] == 255) or (
                                edges[i - 1, j + 1] == 255):
                            edges[i, j] = 255

            # Converter imagem de bordas de volta para imagem PIL
            edges_image = Image.fromarray(edges.astype(np.uint8))

            plt.subplot(1, 2, 1)
            plt.imshow(np_image, cmap='gray')
            plt.title('Imagem original')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(edges_image, cmap='gray')
            plt.title('Bordas - Detector de Canny')
            plt.axis('off')

            plt.show()

    
    def segmentacao_crescimento_regioes(self):
        if self.image:
            imagem_cinza = self.image.convert('L')
            np_imagem = np.array(imagem_cinza)

            x_semente, y_semente = simpledialog.askinteger("Segmentação por Crescimento de Regiões",
                                                          "Digite as coordenadas x e y da semente separadas por espaço:")

            if x_semente is not None and y_semente is not None:
                # Verifica se as coordenadas da semente estão dentro dos limites da imagem
                altura, largura = np_imagem.shape
                if 0 <= x_semente < largura and 0 <= y_semente < altura:
                    limiar = simpledialog.askinteger("Segmentação por Crescimento de Regiões",
                                                     "Digite o valor do limiar de crescimento:")

                    if limiar is not None:
                        # Usando o algoritmo de crescimento de regiões da scikit-image
                        segmentada = segmentation.flood(np_imagem, (y_semente, x_semente), tolerance=limiar, connectivity=1)

                        # Converte a imagem segmentada para a cor preta (0) na imagem final
                        np_imagem[segmentada] = 0

                        imagem_segmentada = Image.fromarray(np_imagem)

                        plt.subplot(1, 2, 1)
                        plt.imshow(imagem_cinza, cmap='gray')
                        plt.title('Imagem original')
                        plt.axis('off')

                        plt.subplot(1, 2, 2)
                        plt.imshow(imagem_segmentada, cmap='gray')
                        plt.title('Segmentação por Crescimento de Regiões')
                        plt.axis('off')

                        plt.show()
                else:
                    messagebox.showerror("Erro", "Coordenadas da semente fora dos limites da imagem!")
    
    def codigo_da_cadeia(self):
        if self.image:
            gray_image = self.image.convert('L')
            np_image = np.array(gray_image)
            height, width = np_image.shape

            # Definir os valores dos vizinhos
            vizinhos = [(1, 0), (1, 1), (0, 1), (-1, 1),
                        (-1, 0), (-1, -1), (0, -1), (1, -1)]

            # Criar uma matriz para armazenar o código da cadeia
            codigo_cadeia = np.zeros((height, width), dtype=int)

            # Encontrar o primeiro pixel com valor diferente de zero
            start_pixel = np.argwhere(np_image != 0)[0]
            current_pixel = tuple(start_pixel)
            current_direction = 0

            # Definir os valores do código da cadeia para cada direção
            codigo_cadeia_direcoes = [0, 1, 2, 3, 4, 5, 6, 7]

            # Gerar o código da cadeia
            for i in range(8 * height * width):
                x, y = current_pixel
                next_direction = (current_direction - 1) % 8

                # Verificar os vizinhos para encontrar o próximo pixel na borda
                for j in range(8):
                    dx, dy = vizinhos[next_direction]
                    next_pixel = (x + dx, y + dy)
                    if np_image[next_pixel] != 0:
                        break
                    next_direction = (next_direction + 1) % 8

                # Calcular o valor do código da cadeia
                codigo_cadeia[x, y] = codigo_cadeia_direcoes[next_direction]
                current_pixel = next_pixel
                current_direction = next_direction

            # Retornar a matriz com o código da cadeia
            return codigo_cadeia


    def esqueletizacao(self):
        if self.image:
            gray_image = self.image.convert('L')
            np_image = np.array(gray_image)

            # Usando o algoritmo de esqueletização do scikit-image
            esqueleto = morphology.skeletonize(np_image > 0)

            esqueleto = (esqueleto.astype(np.uint8) ^ 1) * 255
            esqueleto = esqueleto.astype(np.uint8)
            imagem_esqueletizada = Image.fromarray(esqueleto)

            plt.subplot(1, 2, 1)
            plt.imshow(np_image, cmap='gray')
            plt.title('Imagem original')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(imagem_esqueletizada, cmap='gray')
            plt.title('Esqueletização')
            plt.axis('off')

            plt.show(block=False)  # Mostra o gráfico sem bloquear a interface gráfica
            self.janela.update_idletasks()  # Atualiza a interface gráfica

            # Aguarda até que a janela seja fechada pelo usuário
            while plt.get_fignums():  
                self.janela.update()

    def casamento_por_correlacao(self):
        if self.image:
            gray_image = self.image.convert('L')
            np_image = np.array(gray_image)

            # Carregue o template
            template_path = filedialog.askopenfilename()
            if not template_path:
                return

            template = Image.open(template_path).convert('L')
            np_template = np.array(template)

            # Realize o casamento por correlação
            correlation_result = cv2.matchTemplate(np_image, np_template, cv2.TM_CCOEFF_NORMED)

            plt.subplot(1, 2, 1)
            plt.imshow(np_image, cmap='gray')
            plt.title('Imagem original')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(correlation_result, cmap='gray')
            plt.title('Resultado de Correlação')
            plt.axis('off')

            plt.show()

                
    def mainloop(self):
        self.janela.mainloop()


    def desfazer(self):
        if self.original_image:
            self.image = self.original_image.copy()
            self.show_imagem()

    
    def sair(self):
        sys.exit(0)


    def fechar_janela(self):
        if messagebox.askokcancel("Fechar", "Deseja fechar o aplicativo?"):
            self.janela.destroy()


if __name__ == '__main__':
    processador = ProcessadorImagens()
    processador.mainloop()
