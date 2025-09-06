// Call the dataTables jQuery plugin
$(document).ready(function() {
  $('#dataTable').DataTable({
    dom: 'Blfrtip', // Enables the Buttons extension
        buttons: [
            'excel' // Adds the Export to Excel button
        ]
  });
});
