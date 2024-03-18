import app  # Reemplaza 'your_module_name' con el nombre real de tu módulo

if __name__ == '__main__':
    import os
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8080))
    workers = int(os.environ.get('WORKERS', 1))

    bind_address = f'{host}:{port}'
    worker_class = 'gevent'  # Opcional: Puedes ajustar esto según tus necesidades

    os.system(f'gunicorn -w {workers} -b {bind_address} -k {worker_class} your_module_name:app')