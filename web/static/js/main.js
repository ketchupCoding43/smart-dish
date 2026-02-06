async function fetchInvoice() {
    const response = await fetch("/invoice");
    const data = await response.json();

    document.getElementById("user-info").innerText = `User: ${data.user}`;
    document.getElementById("total").innerText = data.total;
    document.getElementById("status").innerText = data.status;

    const tbody = document.querySelector("#invoice-table tbody");
    tbody.innerHTML = "";

    data.items.forEach(item => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${item.name}</td>
            <td>${item.qty}</td>
            <td>${item.price}</td>
            <td>${item.qty * item.price}</td>
        `;
        tbody.appendChild(row);
    });
}

// Refresh every 2 seconds
setInterval(fetchInvoice, 2000);
fetchInvoice();
