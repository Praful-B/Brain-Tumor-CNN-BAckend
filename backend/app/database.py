import psycopg2

def get_connection() :
    return psycopg2.connect(host = "localhost", database = "brain_tumor_db", user = "brainuser", password = "braintumor")

def fetch_predictions(limits = 15) :

    connect = get_connection()
    curs = connect.cursor()

    curs.execute("SELECT * FROM tumor_predictions ORDER BY created_at DESC LIMIT %s", (limits))

    rows = curs.fetchall()

    curs.close()
    connect.close()

    return [{"id":r[0],"image_name":r[1],"predicted_label":r[2],"confidence":float(r[3]),"created_at":r[4].strftime("%Y-%m-%d %H:%M:%S")}for r in rows]

