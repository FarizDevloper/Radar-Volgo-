import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
import random
import logging
import pywt  # Untuk Wavelet Transform
import cv2  # Untuk Kalman Filter
import requests
import sqlite3  # Untuk menyimpan riwayat penerbangan
import folium  # Untuk pemetaan pesawat

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Fungsi untuk membuat filter bandpass
def butter_bandpass(lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band')

# Fungsi untuk menerapkan filter pada sinyal
def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

# Fungsi untuk mengurangi noise menggunakan wavelet transform
def wavelet_denoise(signal, wavelet='db4', level=5):
    coeff = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal))) * (1/np.sqrt(2))
    coeff[1:] = [pywt.threshold(i, threshold, mode='soft') for i in coeff[1:]]
    return pywt.waverec(coeff, wavelet)

# Fungsi Kalman Filter untuk mengurangi noise
def kalman_filter(data):
    kalman = cv2.KalmanFilter(1, 1, 0)
    kalman.transitionMatrix = np.array([[1]], np.float32)
    kalman.measurementMatrix = np.array([[1]], np.float32)
    kalman.processNoiseCov = np.array([[1e-5]], np.float32)
    kalman.measurementNoiseCov = np.array([[1e-1]], np.float32)
    kalman.errorCovPost = np.array([[1]], np.float32)
    
    filtered_data = []
    for measurement in data:
        kalman.correct(np.array([measurement], np.float32))
        prediction = kalman.predict()
        filtered_data.append(prediction[0])
    
    return np.array(filtered_data)

# Fungsi untuk menghasilkan sinyal radar acak
def generate_random_radar():
    fs = 1000.0  # Frekuensi sampling
    T = 1.0      # Durasi waktu
    lowcut = 10.0
    highcut = 200.0
    f_signal = random.randint(1, 100)
    f_noise = random.randint(100, 500)

    t = np.linspace(0, T, int(T * fs), endpoint=False)
    sinyal_asli = np.sin(2 * np.pi * f_signal * t) + 0.5 * np.sin(2 * np.pi * f_noise * t)

    sinyal_filtered = butter_bandpass_filter(sinyal_asli, lowcut, highcut, fs)
    sinyal_denoised = wavelet_denoise(sinyal_filtered)
    sinyal_filtered_kalman = kalman_filter(sinyal_denoised)

    # Membuat plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(t, sinyal_asli, label='Sinyal Asli dengan Noise Radar')
    ax1.set_title(f'Sinyal Asli dengan Noise Radar Acak\nSignal: {f_signal} Hz, Noise: {f_noise} Hz')
    ax1.set_xlim(0, T)
    ax1.set_ylim(-2, 2)
    ax1.legend()

    ax2.plot(t, sinyal_filtered_kalman, label='Sinyal Setelah Filter (Kalman Denoise)')
    ax2.set_title('Sinyal Setelah Menghilangkan Noise Radar')
    ax2.set_xlim(0, T)
    ax2.set_ylim(-2, 2)
    ax2.legend()

    # Menyimpan plot dalam format gambar
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Fungsi untuk mendapatkan data penerbangan dari ADSB Exchange dengan filter
def get_aircraft_data(speed_threshold=300):
    url = "https://globe.adsbexchange.com/data.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Mengambil pesawat dengan kecepatan di atas threshold
        filtered_aircraft = [
            aircraft for aircraft in data.get('aircraft', [])
            if aircraft.get('vel', 0) > speed_threshold
        ]
        if filtered_aircraft:
            aircraft = filtered_aircraft[0]  # Ambil pesawat pertama yang memenuhi kriteria
            flight_info = f"ID: {aircraft.get('icao24', 'N/A')}\n" \
                          f"Lat: {aircraft.get('lat', 'N/A')}, Lon: {aircraft.get('lon', 'N/A')}\n" \
                          f"Kecepatan: {aircraft.get('vel', 'N/A')}\n" \
                          f"Arah: {aircraft.get('track', 'N/A')}\n"
            return flight_info
        else:
            return f"Tidak ada pesawat dengan kecepatan lebih dari {speed_threshold} knots."
    except requests.RequestException as e:
        logging.error(f"Error fetching aircraft data: {e}")
        return "Terjadi kesalahan saat mengambil data penerbangan."

# Fungsi untuk membuat peta interaktif dengan posisi pesawat
def create_map(lat, lon):
    # Buat peta dengan folium
    map_ = folium.Map(location=[lat, lon], zoom_start=10)
    folium.Marker([lat, lon], popup="Pesawat Terdeteksi").add_to(map_)
    map_file = BytesIO()
    map_.save(map_file, close_file=False)
    map_file.seek(0)
    return map_file

# Fungsi untuk mengirim update penerbangan secara berkala
async def send_aircraft_update(context: CallbackContext):
    chat_id = context.job.context
    flight_info = get_aircraft_data(speed_threshold=350)  # Menambahkan filter kecepatan 350 knots
    await context.bot.send_message(chat_id=chat_id, text=flight_info)
    logging.info(f"Flight update sent to chat_id: {chat_id}")

# Fungsi untuk memulai pemantauan penerbangan
async def monitor_aircraft_live(update: Update, context: CallbackContext):
    if 'job' in context.chat_data:
        context.chat_data['job'].schedule_removal()

    interval = 10
    if context.args:
        try:
            interval = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Masukkan interval waktu yang valid.")
            return

    job = context.job_queue.run_repeating(send_aircraft_update, interval=interval, first=0)
    job.context = update.message.chat_id
    context.chat_data['job'] = job
    await update.message.reply_text(f"Pemantauan penerbangan dimulai dengan interval {interval} detik.")

# Fungsi untuk menghentikan pemantauan penerbangan
async def stop_monitoring(update: Update, context: CallbackContext):
    if 'job' in context.chat_data:
        context.chat_data['job'].schedule_removal()
        del context.chat_data['job']
        await update.message.reply_text("Pemantauan penerbangan dihentikan.")
    else:
        await update.message.reply_text("Tidak ada pemantauan penerbangan yang aktif saat ini.")

# Fungsi untuk mengirimkan pesan saat bot dimulai
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Halo! Saya bot yang dapat memantau penerbangan secara langsung.\n"
        "Gunakan perintah /monitor_aircraft_live <interval detik> untuk memulai pemantauan.\n"
        "Gunakan perintah /stop_monitoring untuk menghentikan pemantauan.\n"
        "Gunakan perintah /plot untuk melihat sinyal radar."
    )

# Fungsi untuk memplot sinyal radar
async def plot(update: Update, context: CallbackContext):
    buf = generate_random_radar()  
    await update.message.reply_photo(photo=buf)
    buf.close()

# Fungsi utama untuk menjalankan bot
def main():
    TOKEN = '7761948010:AAH3YwWZHibjXIXHIAwscpdueuLQj5n-wC0'  # Ganti dengan API Token bot Telegram Anda
    application = Application.builder().token(TOKEN).build()

    # Menambahkan handler untuk perintah /start, /monitor_aircraft_live, /stop_monitoring, dan /plot
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('plot', plot))
    application.add_handler(CommandHandler('monitor_aircraft_live', monitor_aircraft_live))
    application.add_handler(CommandHandler('stop_monitoring', stop_monitoring))

    # Memulai bot dengan polling
    application.run_polling()

if __name__ == '__main__':
    main()
