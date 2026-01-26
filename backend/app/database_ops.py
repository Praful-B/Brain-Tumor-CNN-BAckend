from .database import get_connection

def save_predictions(image_path, label, confidence) :

    connect = get_connection()
    curs = connect.cursor()

    curs.execute("INSERT INTO tumor_predictions (image_name, predicted_label, confidence) VALUES (%s, %s, %s)", (image_path, label, confidence))

    connect.commit()
    curs.close()
    connect.close()
    