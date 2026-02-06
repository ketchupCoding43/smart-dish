class Invoice:
    def __init__(self, user_id):
        self.user_id = user_id
        self.items = []
        self.total = 0
        self.invoice_id = f"INV-{user_id}"

    def add_item(self, name, price):
        self.items.append({"name": name, "price": price})
        self.total += price
